# metrics.py - Lightweight version
from typing import Dict, Any

def calculate_bleu(reference: str, hypothesis: str) -> float:
    """Simple BLEU approximation"""
    try:
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        if not ref_words:
            return 0.0
        matches = sum(1 for w in hyp_words if w in ref_words)
        return matches / max(len(hyp_words), 1)
    except:
        return 0.0


def calculate_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    """Very basic ROUGE-1"""
    try:
        ref = set(reference.lower().split())
        hyp = set(hypothesis.lower().split())
        overlap = len(ref.intersection(hyp))
        precision = overlap / len(hyp) if hyp else 0
        recall = overlap / len(ref) if ref else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return {"rouge1": f1, "rouge2": 0.0, "rougeL": f1}
    except:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def calculate_basic_metrics(original: str, translation: str) -> Dict[str, Any]:
    return {
        "length_ratio": len(translation.split()) / max(len(original.split()), 1),
        "char_count": len(translation),
        "word_count": len(translation.split())
    }