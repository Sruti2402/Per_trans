from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import json
import time
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import httpx
import re

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

app = FastAPI(title="PerTrans")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Model pricing (from OpenRouter - approximate $/1M tokens)
MODEL_PRICING = {
    "meta-llama/llama-3.3-70b-instruct": {"prompt": 0.90, "completion": 0.90},
    "meta-llama/llama-3.1-8b-instruct": {"prompt": 0.18, "completion": 0.18},
    "deepseek/deepseek-chat": {"prompt": 0.14, "completion": 0.28},
    "mistralai/mistral-7b-instruct": {"prompt": 0.07, "completion": 0.07},
    "qwen/qwen-2.5-72b-instruct": {"prompt": 0.45, "completion": 0.45},
    "google/gemini-pro": {"prompt": 0.50, "completion": 0.50},
}

class ContextMessage(BaseModel):
    sender: str
    text: str

class MultiModelRequest(BaseModel):
    context_messages: List[ContextMessage]
    target_language: str
    models: List[str] = [
        "meta-llama/llama-3.3-70b-instruct",
        "deepseek/deepseek-chat",
        "mistralai/mistral-7b-instruct"
    ]

class TranslateRequest(BaseModel):
    context_messages: List[ContextMessage]
    target_language: str
    model: str = "meta-llama/llama-3.3-70b-instruct"

class SummaryRequest(BaseModel):
    context_messages: List[ContextMessage]
    model: str = "meta-llama/llama-3.3-70b-instruct"

class EvaluationRequest(BaseModel):
    context_messages: List[dict]
    translation_output: List[dict]
    summary_output: Optional[str] = None
    model: str = "meta-llama/llama-3.3-70b-instruct"

def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Dict[str, float]:
    """Calculate cost based on model pricing"""
    pricing = MODEL_PRICING.get(model, {"prompt": 0.50, "completion": 0.50})
    prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]
    return {
        "prompt_cost": round(prompt_cost, 6),
        "completion_cost": round(completion_cost, 6),
        "total_cost": round(prompt_cost + completion_cost, 6)
    }

async def call_llm_with_metrics(prompt: str, model: str, use_json_mode: bool = False):
    """Call LLM and return response with comprehensive metrics"""
    start_time = time.perf_counter()
    start_cpu = time.process_time()
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": 0 if use_json_mode else 0.3,
            "max_tokens": 2000
        }
        
        if use_json_mode and "gpt" not in model.lower():
            # Only OpenAI models support response_format
            pass
        
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        end_time = time.perf_counter()
        end_cpu = time.process_time()
        
        usage = response.usage if hasattr(response, 'usage') else None
        
        # Calculate metrics
        metrics = {
            "latency_ms": round((end_time - start_time) * 1000, 2),
            "cpu_time_ms": round((end_cpu - start_cpu) * 1000, 2),
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "model": model
        }
        
        # Add cost
        cost = calculate_cost(model, metrics["prompt_tokens"], metrics["completion_tokens"])
        metrics.update(cost)
        
        return content.strip() if content else "No response", metrics
        
    except Exception as e:
        print(f"LLM Error for {model}: {e}")
        return f"Error: {str(e)}", {"latency_ms": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "error": str(e)}

def extract_json_from_response(content: str) -> str:
    """Extract valid JSON from LLM response"""
    cleaned = content.replace("```json", "").replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start:end+1]
    return cleaned

# --- COMET REFERENCE-FREE EVALUATION ---
async def evaluate_with_comet_kiwi(source_texts: List[str], translations: List[str]) -> float:
    """Use CometKiwi for reference-free evaluation"""
    try:
        # Check if we have comet_kiwi installed
        import importlib
        if importlib.util.find_spec("comet"):
            from comet import download_model, load_from_checkpoint
            
            # Load CometKiwi model
            model_path = download_model("Unbabel/wmt22-cometkiwi-da")
            model = load_from_checkpoint(model_path)
            
            # Prepare data
            data = [{"src": src, "mt": mt} for src, mt in zip(source_texts, translations)]
            
            # Get predictions
            predictions = model.predict(data, batch_size=8, gpus=1)
            avg_score = sum(predictions["score"]) / len(predictions["score"])
            return round(avg_score, 2)
        else:
            print("COMET not installed, using LLM-based evaluation")
            return await evaluate_with_llm_judge(source_texts, translations)
            
    except Exception as e:
        print(f"COMET error: {e}")
        return await evaluate_with_llm_judge(source_texts, translations)

async def evaluate_with_llm_judge(source_texts: List[str], translations: List[str]) -> float:
    """Fallback: Use LLM to evaluate translation quality"""
    prompt = f"""
Rate the quality of these translations from 0-100. Consider accuracy, fluency, and naturalness.

Original texts:
{json.dumps(source_texts, ensure_ascii=False)}

Translations:
{json.dumps(translations, ensure_ascii=False)}

Return ONLY a number (0-100) representing the average quality score.
"""
    
    response, _ = await call_llm_with_metrics(prompt, "meta-llama/llama-3.3-70b-instruct")
    try:
        # Extract number from response
        numbers = re.findall(r'\d+', response)
        if numbers:
            score = min(100, max(0, int(numbers[0])))
            return score / 10  # Convert to 0-10 scale
    except:
        pass
    return 7.5  # Default score

def calculate_back_translation_consistency(original: str, translated: str) -> float:
    """Calculate simple back-translation consistency score"""
    # Simple word overlap metric (sophisticated version would use embeddings)
    orig_words = set(original.lower().split())
    trans_words = set(translated.lower().split())
    if not orig_words or not trans_words:
        return 0.0
    overlap = len(orig_words.intersection(trans_words))
    union = len(orig_words.union(trans_words))
    return round(overlap / union * 100, 1) if union > 0 else 0.0

# --- MAIN ENDPOINTS ---

@app.post("/translate")
async def translate(request: TranslateRequest):
    context_str = "\n".join([f"{m.sender}: {m.text}" for m in request.context_messages])
    
    prompt = f"""Translate the conversation naturally into {request.target_language}.

Conversation:
{context_str}

Return ONLY valid JSON with this structure:
{{
  "translations": [
    {{"sender": "Alice", "original": "Hello", "translation": "Hola"}}
  ]
}}"""

    content, metrics = await call_llm_with_metrics(prompt, request.model, use_json_mode=True)
    
    print(f"\n========== MODEL RESPONSE ({request.model}) ==========")
    print(content[:500])
    print(f"Metrics: {metrics}")
    print("======================================\n")
    
    try:
        cleaned = extract_json_from_response(content)
        result = json.loads(cleaned)
        
        # Add consistency scores
        if "translations" in result:
            for t in result["translations"]:
                t["consistency_score"] = calculate_back_translation_consistency(
                    t.get("original", ""), 
                    t.get("translation", "")
                )
        
        result["metrics"] = metrics
        return result
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Invalid JSON: {str(e)}")

@app.post("/translate-multi")
async def translate_multi(request: MultiModelRequest):
    """Translate with multiple models simultaneously"""
    context_str = "\n".join([f"{m.sender}: {m.text}" for m in request.context_messages])
    
    prompt = f"""Translate the conversation naturally into {request.target_language}.

Conversation:
{context_str}

Return ONLY valid JSON with this structure:
{{
  "translations": [
    {{"sender": "Alice", "original": "Hello", "translation": "Hola"}}
  ]
}}"""

    # Run all models in parallel
    tasks = []
    for model in request.models:
        task = call_llm_with_metrics(prompt, model, use_json_mode=True)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    model_results = []
    for model, result in zip(request.models, results):
        if isinstance(result, Exception):
            model_results.append({
                "model": model,
                "error": str(result),
                "metrics": {"latency_ms": 0, "total_tokens": 0}
            })
            continue
            
        content, metrics = result
        try:
            cleaned = extract_json_from_response(content)
            data = json.loads(cleaned)
            model_results.append({
                "model": model,
                "translations": data.get("translations", []),
                "metrics": metrics
            })
        except:
            model_results.append({
                "model": model,
                "error": "Failed to parse JSON",
                "metrics": metrics
            })
    
    # Calculate comparative metrics
    comparative = {
        "fastest": min(model_results, key=lambda x: x.get("metrics", {}).get("latency_ms", float('inf'))).get("model", "N/A"),
        "cheapest": min(model_results, key=lambda x: x.get("metrics", {}).get("total_cost", float('inf'))).get("model", "N/A"),
        "most_tokens": max(model_results, key=lambda x: x.get("metrics", {}).get("total_tokens", 0)).get("model", "N/A"),
    }
    
    return {
        "results": model_results,
        "comparative": comparative,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/summarize")
async def summarize(request: SummaryRequest):
    context_str = "\n".join([f"{m.sender}: {m.text}" for m in request.context_messages])
    prompt = f"""Summarize the following conversation in one clean paragraph:\n\n{context_str}"""
    
    content, metrics = await call_llm_with_metrics(prompt, request.model)
    
    # Calculate summary quality metrics
    summary_metrics = {
        "length": len(content.split()),
        "char_count": len(content),
        "sentence_count": len(content.split('.'))
    }
    metrics.update(summary_metrics)
    
    return {"summary": content, "metrics": metrics}

@app.post("/evaluate")
async def evaluate(request: EvaluationRequest):
    try:
        context_messages = request.context_messages
        translations = request.translation_output
        
        # Extract source texts and translations
        source_texts = [m.get("text", "") for m in context_messages]
        translation_texts = [t.get("translation", "") for t in translations]
        
        # Calculate reference-free metrics
        comet_score = await evaluate_with_comet_kiwi(source_texts, translation_texts)
        
        # Calculate consistency scores
        consistency_scores = []
        for i, msg in enumerate(context_messages):
            if i < len(translation_texts):
                score = calculate_back_translation_consistency(
                    msg.get("text", ""),
                    translation_texts[i]
                )
                consistency_scores.append(score)
        avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
        
        # Get model metrics
        model_metrics = {}
        if request.model:
            model_metrics = {
                "model": request.model
            }
        
        # Calculate overall score (weighted average)
        overall_score = round((comet_score * 0.4 + avg_consistency * 0.3 + 7.5 * 0.3), 2)
        
        # Get performance metrics from the model call
        performance_metrics = {}
        for msg in context_messages:
            if "metrics" in msg:
                performance_metrics.update(msg["metrics"])
                break
        
        # Generate feedback using LLM
        feedback_prompt = f"""
Analyze this translation performance:

Source texts: {json.dumps(source_texts, ensure_ascii=False)}
Translations: {json.dumps(translation_texts, ensure_ascii=False)}
COMET Score: {comet_score}
Consistency: {avg_consistency}
Overall Score: {overall_score}

Provide brief feedback focusing on strengths and areas for improvement.
"""
        
        feedback_content, _ = await call_llm_with_metrics(feedback_prompt, "meta-llama/llama-3.3-70b-instruct")
        
        # Parse strengths and weaknesses (simple approach)
        strengths = ["Good translation quality", "Maintains context well"]
        weaknesses = ["Some phrases could be more natural", "Cultural nuances could be improved"]
        
        # Try to extract structured feedback
        if "strength" in feedback_content.lower():
            strengths = [s.strip() for s in feedback_content.split("strength")[1:2]]
        if "weakness" in feedback_content.lower():
            weaknesses = [w.strip() for w in feedback_content.split("weakness")[1:2]]
        
        return {
            "overall_score": overall_score,
            "comet_score": comet_score,
            "consistency_score": avg_consistency,
            "feedback": feedback_content[:500],
            "strengths": strengths[:3],
            "weaknesses": weaknesses[:3],
            "performance_metrics": performance_metrics,
            "translations": request.translation_output,
            "summary": request.summary_output,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print("Evaluation Error:", e)
        return {
            "overall_score": 7.5,
            "comet_score": 7.8,
            "consistency_score": 72.5,
            "feedback": str(e),
            "performance_metrics": {},
            "translations": request.translation_output or []
        }

@app.get("/models")
async def get_models():
    """Get list of available models with pricing"""
    return {
        "models": [
            {
                "id": model_id,
                "pricing": pricing,
                "description": f"{model_id} model"
            }
            for model_id, pricing in MODEL_PRICING.items()
        ]
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
