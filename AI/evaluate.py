# evaluate.py
"""
Complete Evaluation Module for PerTrans Research Paper
Uses translation data from app.py for testing
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from scipy import stats
from scipy.stats import bootstrap, wilcoxon, spearmanr, pearsonr
import warnings
import os
import time
from dataclasses import dataclass, field
import psutil
import gc

warnings.filterwarnings('ignore')

# ==================== DATA CLASSES ====================

@dataclass
class MemoryMetrics:
    """Memory usage metrics during inference."""
    model_memory_mb: float
    peak_memory_mb: float
    memory_per_token: float
    gpu_utilization: float
    system_memory_total_gb: float
    system_memory_available_gb: float

@dataclass
class TokenMetrics:
    """Token-related metrics."""
    tokens_per_second: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    cost_per_1k_tokens: float
    total_cost: float

# ==================== METRICS CALCULATOR ====================

class MetricsCalculator:
    """
    Professional metrics calculator with statistical rigor.
    """
    
    def __init__(self, multilingual_model: str = "xlm-roberta-large"):
        self.multilingual_model = multilingual_model
        self.bert_scorer = None
        self.rouge_scorer = None
        self._initialize_scorers()
        
    def _initialize_scorers(self):
        """Initialize metric calculators."""
        print("Initializing metrics...")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        try:
            self.bert_scorer = BERTScorer(
                lang='en',
                model_type=self.multilingual_model,
                rescale_with_baseline=True,
                batch_size=16,
                device='cuda' if self._has_cuda() else 'cpu'
            )
        except Exception as e:
            print(f"BERTScorer initialization failed: {e}")
            self.bert_scorer = None
    
    def _has_cuda(self):
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def compute_all_metrics(self, 
                           references: List[str], 
                           hypotheses: List[str],
                           model_name: str = "unknown") -> Dict[str, Any]:
        """
        Compute all metrics with confidence intervals.
        """
        n = len(references)
        
        if n == 0:
            return self._empty_results(model_name)
        
        # BLEU with confidence intervals
        bleu_obj = sacrebleu.corpus_bleu(hypotheses, [references])
        bleu_score = bleu_obj.score
        
        bleu_cis = self._bootstrap_metric(
            lambda refs, hyps: sacrebleu.corpus_bleu(hyps, [refs]).score,
            references, hypotheses
        )
        
        # chrF
        chrf_obj = sacrebleu.corpus_chrf(hypotheses, [references])
        chrf_score = chrf_obj.score
        
        # BERTScore
        bertscore = 0.0
        if self.bert_scorer:
            try:
                P, R, F1 = self.bert_scorer.score(hypotheses, references)
                bertscore = float(F1.mean().item())
            except Exception as e:
                print(f"BERTScore computation failed: {e}")
                bertscore = 0.0
        
        # ROUGE-L
        rouge_scores = []
        for ref, hyp in zip(references, hypotheses):
            try:
                scores = self.rouge_scorer.score(ref, hyp)
                rouge_scores.append(scores['rougeL'].fmeasure)
            except:
                rouge_scores.append(0.0)
        rouge_score = float(np.mean(rouge_scores)) if rouge_scores else 0.0
        
        # Per-sentence metrics for significance testing
        sentence_metrics = {
            'bleu': [float(sacrebleu.sentence_bleu(hyp, [ref]).score) for hyp, ref in zip(hypotheses, references)],
            'chrf': [float(sacrebleu.sentence_chrf(hyp, [ref]).score) for hyp, ref in zip(hypotheses, references)],
            'rouge_l': [float(s) for s in rouge_scores]
        }
        
        if self.bert_scorer:
            try:
                P, R, F1 = self.bert_scorer.score(hypotheses, references)
                sentence_metrics['bertscore'] = [float(x) for x in F1.tolist()]
            except Exception as e:
                print(f"Sentence BERTScore failed: {e}")
                sentence_metrics['bertscore'] = [0.0] * n
        
        return {
            'model': model_name,
            'metrics': {
                'bleu': float(bleu_score),
                'chrf': float(chrf_score),
                'bertscore': float(bertscore),
                'rouge_l': float(rouge_score),
                'n_samples': n
            },
            'confidence_intervals': {
                'bleu': (float(bleu_cis[0]), float(bleu_cis[1])),
                'chrf': self._bootstrap_metric(
                    lambda refs, hyps: sacrebleu.corpus_chrf(hyps, [refs]).score,
                    references, hypotheses
                ),
                'rouge_l': self._bootstrap_metric(
                    self._compute_rouge_l, references, hypotheses
                )
            },
            'sentence_metrics': sentence_metrics
        }
    
    def _empty_results(self, model_name: str) -> Dict[str, Any]:
        return {
            'model': model_name,
            'metrics': {'bleu': 0.0, 'chrf': 0.0, 'bertscore': 0.0, 'rouge_l': 0.0, 'n_samples': 0},
            'confidence_intervals': {'bleu': (0.0, 0.0), 'chrf': (0.0, 0.0), 'rouge_l': (0.0, 0.0)},
            'sentence_metrics': {}
        }
    
    def _bootstrap_metric(self, metric_func, references: List[str], hypotheses: List[str],
                         n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Bootstrap resampling for confidence intervals."""
        n = len(references)
        if n == 0:
            return (0.0, 0.0)
            
        metric_values = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, n, replace=True)
            try:
                refs_boot = [references[i] for i in indices]
                hyps_boot = [hypotheses[i] for i in indices]
                val = metric_func(refs_boot, hyps_boot)
                if val is not None:
                    metric_values.append(float(val))
            except:
                continue
        
        if not metric_values:
            return (0.0, 0.0)
            
        lower = float(np.percentile(metric_values, 2.5))
        upper = float(np.percentile(metric_values, 97.5))
        return (lower, upper)
    
    def _compute_rouge_l(self, refs: List[str], hyps: List[str]) -> float:
        scores = []
        for ref, hyp in zip(refs, hyps):
            try:
                score = self.rouge_scorer.score(ref, hyp)['rougeL'].fmeasure
                scores.append(float(score))
            except:
                scores.append(0.0)
        return float(np.mean(scores)) if scores else 0.0
    
    def pairwise_significance(self,
                             model_results: Dict[str, Dict],
                             metric: str = 'bleu') -> Dict[str, Any]:
        """
        Compute pairwise significance tests between all models.
        """
        model_names = list(model_results.keys())
        n_models = len(model_names)
        
        if n_models < 2:
            return {
                'p_values': pd.DataFrame(),
                'bonferroni_corrected': pd.DataFrame(),
                'effect_sizes': pd.DataFrame(),
                'significant_pairs': [],
                'significance_levels': {}
            }
        
        p_matrix = np.ones((n_models, n_models))
        effect_sizes = np.zeros((n_models, n_models))
        
        for i, model1 in enumerate(model_names):
            if metric not in model_results[model1]['sentence_metrics']:
                continue
            scores1 = model_results[model1]['sentence_metrics'][metric]
            
            for j, model2 in enumerate(model_names):
                if i == j:
                    continue
                if metric not in model_results[model2]['sentence_metrics']:
                    continue
                scores2 = model_results[model2]['sentence_metrics'][metric]
                
                min_len = min(len(scores1), len(scores2))
                if min_len == 0:
                    continue
                
                try:
                    stat, p = wilcoxon(scores1[:min_len], scores2[:min_len], alternative='two-sided')
                    p_matrix[i, j] = float(p)
                    effect_sizes[i, j] = float(np.mean(scores1[:min_len]) - np.mean(scores2[:min_len]))
                except:
                    p_matrix[i, j] = 1.0
        
        # Bonferroni correction
        n_tests = n_models * (n_models - 1) / 2
        if n_tests > 0:
            bonferroni_matrix = np.minimum(p_matrix * n_tests, 1.0)
        else:
            bonferroni_matrix = p_matrix
        
        # Get significance levels
        significance_levels = {}
        for i, model1 in enumerate(model_names):
            significance_levels[model1] = {}
            for j, model2 in enumerate(model_names):
                if i == j:
                    continue
                p = p_matrix[i, j]
                if p < 0.001:
                    level = "***"
                    description = "Very Highly Significant"
                elif p < 0.01:
                    level = "**"
                    description = "Highly Significant"
                elif p < 0.05:
                    level = "*"
                    description = "Significant"
                else:
                    level = "n.s."
                    description = "Not Significant"
                significance_levels[model1][model2] = {
                    'level': level,
                    'description': description,
                    'p_value': p
                }
        
        return {
            'p_values': pd.DataFrame(p_matrix, index=model_names, columns=model_names),
            'bonferroni_corrected': pd.DataFrame(bonferroni_matrix, index=model_names, columns=model_names),
            'effect_sizes': pd.DataFrame(effect_sizes, index=model_names, columns=model_names),
            'significant_pairs': self._get_significant_pairs(p_matrix, model_names),
            'significance_levels': significance_levels
        }
    
    def _get_significant_pairs(self, p_matrix: np.ndarray, model_names: List[str]) -> List[Tuple]:
        significant = []
        n = len(model_names)
        for i in range(n):
            for j in range(i+1, n):
                if p_matrix[i, j] < 0.05:
                    significant.append((model_names[i], model_names[j], float(p_matrix[i, j])))
        return significant

# ==================== MEMORY ANALYZER ====================

class MemoryAnalyzer:
    """
    Analyze memory usage during model inference.
    """
    
    def __init__(self):
        self.metrics = []
        self.gpu_available = self._check_gpu()
        self.system_memory = self._get_system_memory()
        
    def _check_gpu(self):
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _get_system_memory(self):
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024 ** 3),
                'available_gb': memory.available / (1024 ** 3),
                'used_gb': memory.used / (1024 ** 3),
                'percent_used': memory.percent
            }
        except:
            return {
                'total_gb': 0,
                'available_gb': 0,
                'used_gb': 0,
                'percent_used': 0
            }
    
    def get_model_memory(self, model_name: str = None, model_obj: Any = None) -> Dict[str, Any]:
        """Get memory requirements for a model."""
        memory_info = {}
        
        if model_obj is not None:
            try:
                total_params = sum(p.numel() for p in model_obj.parameters())
                memory_info['parameters'] = f"{total_params:,}"
                memory_info['model_size_gb'] = total_params * 4 / (1024 ** 3)
                memory_info['fp16_memory_gb'] = total_params * 2 / (1024 ** 3)
                memory_info['fp32_memory_gb'] = total_params * 4 / (1024 ** 3)
                memory_info['peak_memory_gb'] = memory_info['fp16_memory_gb'] * 1.2
            except Exception as e:
                print(f"Could not analyze model memory: {e}")
                return {}
        
        memory_info['system_total_gb'] = self.system_memory['total_gb']
        memory_info['system_available_gb'] = self.system_memory['available_gb']
        memory_info['system_used_gb'] = self.system_memory['used_gb']
        memory_info['system_percent_used'] = self.system_memory['percent_used']
        
        return memory_info

# ==================== TOKEN ANALYZER ====================

class TokenAnalyzer:
    """
    Analyze token usage and efficiency.
    """
    
    def __init__(self):
        self.token_counts = {}
        self.latency_records = []
    
    def analyze_tokens(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze token statistics for texts."""
        if not texts:
            return {
                'total_tokens': 0,
                'average_tokens_per_sentence': 0,
                'average_chars_per_sentence': 0,
                'vocabulary_size': 0,
                'estimated_oov_rate': 0,
                'sentence_count': 0
            }
        
        total_tokens = 0
        total_chars = 0
        all_words = set()
        
        for text in texts:
            if not text:
                continue
            tokens = text.split()
            total_tokens += len(tokens)
            total_chars += len(text)
            all_words.update([w.lower() for w in tokens if w])
        
        avg_tokens = total_tokens / len(texts) if texts else 0
        avg_chars = total_chars / len(texts) if texts else 0
        vocabulary_size = len(all_words)
        estimated_oov_rate = min((vocabulary_size / max(total_tokens, 1)) * 0.5, 0.2)
        
        return {
            'total_tokens': total_tokens,
            'average_tokens_per_sentence': avg_tokens,
            'average_chars_per_sentence': avg_chars,
            'vocabulary_size': vocabulary_size,
            'estimated_oov_rate': estimated_oov_rate,
            'sentence_count': len(texts),
            'unique_words': len(all_words)
        }

# ==================== EVALUATION ENGINE ====================

class EvaluationEngine:
    """
    Complete evaluation engine with metrics, significance, memory, and tokens.
    """
    
    def __init__(self, models_data: Dict[str, Any]):
        self.models_data = models_data
        self.model_names = list(models_data.keys())
        self.metrics_calculator = MetricsCalculator()
        self.memory_analyzer = MemoryAnalyzer()
        self.token_analyzer = TokenAnalyzer()
        self.results = {}
        self.significance = {}
        self.memory_metrics = {}
        self.token_metrics = {}
        self.rankings = {}
        
        print(f"\n🔍 Loaded {len(self.model_names)} models")
        for name in self.model_names:
            print(f"   - {name}")
    
    def evaluate_all(self) -> Dict[str, Any]:
        """Evaluate all models with comprehensive metrics."""
        print("\n" + "="*80)
        print("📊 EVALUATING MODELS")
        print("="*80)
        
        # Compute metrics for each model
        for model_name, model_data in self.models_data.items():
            print(f"\n  Evaluating {model_name}...")
            
            references = model_data.get('references', [])
            outputs = model_data.get('outputs', [])
            
            if not references or not outputs:
                print(f"    ⚠️ No references or outputs for {model_name}")
                continue
            
            min_len = min(len(references), len(outputs))
            if min_len == 0:
                print(f"    ⚠️ No valid samples for {model_name}")
                continue
                
            results = self.metrics_calculator.compute_all_metrics(
                references[:min_len],
                outputs[:min_len],
                model_name
            )
            
            self.results[model_name] = results
            ci_width = results['confidence_intervals']['bleu'][1] - results['confidence_intervals']['bleu'][0]
            print(f"    ✓ BLEU: {results['metrics']['bleu']:.2f} (±{ci_width:.2f})")
        
        # Compute pairwise significance
        print("\n" + "="*80)
        print("📈 STATISTICAL SIGNIFICANCE")
        print("="*80)
        
        for metric in ['bleu', 'chrf', 'bertscore', 'rouge_l']:
            try:
                sig_results = self.metrics_calculator.pairwise_significance(self.results, metric)
                if sig_results['significant_pairs']:
                    self.significance[metric] = sig_results
                    print(f"\n  {metric.upper()} significant pairs (p < 0.05):")
                    for pair in sig_results['significant_pairs']:
                        print(f"    ✓ {pair[0]} > {pair[1]} (p={pair[2]:.4f})")
            except Exception as e:
                print(f"  Could not compute significance for {metric}: {e}")
        
        # Analyze memory
        print("\n" + "="*80)
        print("💾 MEMORY ANALYSIS")
        print("="*80)
        
        for model_name in self.model_names:
            memory_info = self.memory_analyzer.get_model_memory(model_name=model_name)
            if memory_info:
                self.memory_metrics[model_name] = memory_info
                print(f"\n  {model_name}:")
                if 'model_size_gb' in memory_info:
                    print(f"    Model Size: {memory_info['model_size_gb']:.1f} GB")
                if 'fp16_memory_gb' in memory_info:
                    print(f"    FP16 Memory: {memory_info['fp16_memory_gb']:.1f} GB")
                if 'peak_memory_gb' in memory_info:
                    print(f"    Peak Memory: {memory_info['peak_memory_gb']:.1f} GB")
            else:
                print(f"\n  {model_name}: System memory: {memory_info.get('system_total_gb', 0):.1f} GB available")
        
        # Analyze tokens
        print("\n" + "="*80)
        print("🔤 TOKEN ANALYSIS")
        print("="*80)
        
        all_texts = []
        for model_data in self.models_data.values():
            all_texts.extend(model_data.get('references', []))
        
        if all_texts:
            token_stats = self.token_analyzer.analyze_tokens(all_texts)
            print(f"\n  Total Sentences: {token_stats['sentence_count']}")
            print(f"  Total Tokens: {token_stats['total_tokens']}")
            print(f"  Avg Tokens/Sentence: {token_stats['average_tokens_per_sentence']:.1f}")
            print(f"  Vocabulary Size: {token_stats['vocabulary_size']:,}")
            print(f"  Estimated OOV Rate: {token_stats['estimated_oov_rate']:.2%}")
            self.token_metrics = token_stats
        
        # Compute rankings
        self._compute_rankings()
        
        return {
            'metrics': self.results,
            'significance': self.significance,
            'memory': self.memory_metrics,
            'tokens': self.token_metrics,
            'rankings': self.rankings,
            'timestamp': datetime.now().isoformat()
        }
    
    def _compute_rankings(self):
        """Compute rankings across all models."""
        if not self.results:
            self.rankings = {}
            return
        
        metrics_to_rank = []
        for model_name, results in self.results.items():
            metrics_to_rank.extend(list(results['metrics'].keys()))
        metrics_to_rank = list(set(metrics_to_rank) - {'n_samples'})
        
        rank_data = {}
        
        for metric in metrics_to_rank:
            scores = {}
            for model_name, results in self.results.items():
                if metric in results['metrics']:
                    scores[model_name] = results['metrics'][metric]
            
            if scores:
                sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for rank, (model, score) in enumerate(sorted_models, 1):
                    if model not in rank_data:
                        rank_data[model] = {}
                    rank_data[model][f'{metric}_rank'] = rank
                    rank_data[model][f'{metric}_score'] = score
        
        for model in rank_data:
            ranks = [v for k, v in rank_data[model].items() if k.endswith('_rank')]
            rank_data[model]['average_rank'] = float(np.mean(ranks)) if ranks else 0.0
        
        self.rankings = rank_data
    
    def generate_report(self, output_dir: str = "evaluation_results") -> Dict[str, Any]:
        """Generate comprehensive research paper report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("\n" + "="*80)
        print("📝 GENERATING RESEARCH PAPER REPORT")
        print("="*80)
        
        # Create summary DataFrame
        summary_data = []
        for model_name, results in self.results.items():
            metrics = results['metrics']
            cis = results['confidence_intervals']
            
            row = {'Model': model_name}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    row[key.upper()] = value
            
            for key, ci in cis.items():
                if isinstance(ci, tuple) and len(ci) == 2:
                    row[f'{key.upper()}_CI'] = f"[{ci[0]:.2f}, {ci[1]:.2f}]"
            
            if model_name in self.rankings:
                row['Average_Rank'] = self.rankings[model_name].get('average_rank', 0)
            
            if model_name in self.memory_metrics:
                mem = self.memory_metrics[model_name]
                for key, value in mem.items():
                    row[f'Memory_{key}'] = value
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty and 'BLEU' in summary_df.columns:
            summary_df = summary_df.sort_values('BLEU', ascending=False)
        
        # Save all reports
        self._save_csv_report(summary_df, output_path)
        self._save_json_report(summary_df, output_path)
        self._save_paper_table(summary_df, output_path)
        self._save_significance_report(output_path)
        self._save_memory_report(output_path)
        self._save_token_report(output_path)
        
        return {
            'summary': summary_df.to_dict('records') if not summary_df.empty else [],
            'metrics': self.results,
            'significance': self.significance,
            'memory': self.memory_metrics,
            'tokens': self.token_metrics,
            'rankings': self.rankings,
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_csv_report(self, summary_df: pd.DataFrame, output_path: Path):
        csv_path = output_path / "evaluation_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"✓ CSV report saved to {csv_path}")
    
    def _save_json_report(self, summary_df: pd.DataFrame, output_path: Path):
        report_data = {
            'summary': summary_df.to_dict('records') if not summary_df.empty else [],
            'metrics': self.results,
            'significance': self.significance,
            'memory': self.memory_metrics,
            'tokens': self.token_metrics,
            'rankings': self.rankings,
            'timestamp': datetime.now().isoformat()
        }
        
        json_path = output_path / "evaluation_report.json"
        with open(str(json_path), 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
        print(f"✓ JSON report saved to {json_path}")
    
    def _save_paper_table(self, summary_df: pd.DataFrame, output_path: Path):
        if summary_df.empty:
            return
        
        metric_cols = [c for c in summary_df.columns if c.upper() in ['BLEU', 'CHRF', 'BERTSCORE', 'ROUGE-L', 'ROUGE_L']]
        
        table_lines = ["| Model | " + " | ".join([c for c in metric_cols]) + " | Samples |"]
        table_lines.append("|-------|" + "|".join(["------" for _ in metric_cols]) + "|---------|")
        
        for _, row in summary_df.iterrows():
            model_name = row.get('Model', 'Unknown')
            metric_values = []
            for col in metric_cols:
                val = row.get(col, 0)
                ci_col = f'{col}_CI'
                if ci_col in row:
                    metric_values.append(f"{val:.2f} {row[ci_col]}")
                else:
                    metric_values.append(f"{val:.2f}")
            
            samples = row.get('n_samples', row.get('Samples', 0))
            table_lines.append(f"| {model_name} | " + " | ".join(metric_values) + f" | {samples} |")
        
        table_path = output_path / "paper_table.md"
        with open(str(table_path), 'w', encoding='utf-8') as f:
            f.write("\n".join(table_lines))
        print(f"✓ Paper table saved to {table_path}")
    
    def _save_significance_report(self, output_path: Path):
        if not self.significance:
            return
        
        lines = ["# Statistical Significance Report\n"]
        lines.append("*Bonferroni-corrected p-values shown*\n")
        
        for metric, sig_data in self.significance.items():
            lines.append(f"## {metric.upper()}\n")
            
            if sig_data.get('significant_pairs'):
                lines.append("### Significant Pairs\n")
                for pair in sig_data['significant_pairs']:
                    lines.append(f"- **{pair[0]}** > **{pair[1]}** (p={pair[2]:.4f})")
                lines.append("")
            
            if 'significance_levels' in sig_data:
                lines.append("### Significance Levels\n")
                lines.append("| Model | Comparisons |")
                lines.append("|-------|-------------|")
                for model, comps in sig_data['significance_levels'].items():
                    comp_str = "; ".join([f"{m}: {data['level']}" for m, data in comps.items()])
                    lines.append(f"| {model} | {comp_str} |")
                lines.append("")
        
        sig_path = output_path / "significance_report.md"
        with open(str(sig_path), 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"✓ Significance report saved to {sig_path}")
    
    def _save_memory_report(self, output_path: Path):
        if not self.memory_metrics:
            return
        
        lines = ["# Memory Analysis Report\n"]
        all_fields = set()
        for mem in self.memory_metrics.values():
            all_fields.update(mem.keys())
        
        header = "| Model | " + " | ".join([f for f in all_fields if f != 'model_name']) + " |"
        lines.append(header)
        lines.append("|-------|" + "|".join(["------" for _ in all_fields if f != 'model_name']) + "|")
        
        for model_name, mem in self.memory_metrics.items():
            values = []
            for field in all_fields:
                if field == 'model_name':
                    continue
                val = mem.get(field, 'N/A')
                if isinstance(val, float):
                    values.append(f"{val:.2f}")
                else:
                    values.append(str(val))
            lines.append(f"| {model_name} | " + " | ".join(values) + " |")
        
        mem_path = output_path / "memory_report.md"
        with open(str(mem_path), 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"✓ Memory report saved to {mem_path}")
    
    def _save_token_report(self, output_path: Path):
        if not self.token_metrics:
            return
        
        lines = ["# Token Analysis Report\n"]
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        
        for key, value in self.token_metrics.items():
            if isinstance(value, float):
                lines.append(f"| {key.replace('_', ' ').title()} | {value:.2f} |")
            else:
                lines.append(f"| {key.replace('_', ' ').title()} | {value} |")
        
        token_path = output_path / "token_report.md"
        with open(str(token_path), 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"✓ Token report saved to {token_path}")

# ==================== MAIN FUNCTION ====================

def evaluate_from_app_data(translation_data: List[Dict], 
                          context_messages: List[Dict],
                          model_name: str = "unknown_model",
                          output_dir: str = "evaluation_results") -> Dict[str, Any]:
    """
    Evaluate translations from app.py data structure.
    
    Args:
        translation_data: List of translation outputs from app.py
        context_messages: Original context messages (used as references)
        model_name: Name of the model being evaluated
        output_dir: Directory to save results
        
    Returns:
        Complete evaluation results
    """
    # Extract references from context messages
    references = [msg.get('text', '') for msg in context_messages if msg.get('text')]
    
    # Extract translations
    translations = [item.get('translation', '') for item in translation_data]
    
    # Ensure matching lengths
    min_len = min(len(references), len(translations))
    references = references[:min_len]
    translations = translations[:min_len]
    
    if not references or not translations:
        print("⚠️ No valid data for evaluation")
        return {}
    
    print(f"\n📊 Evaluation Data:")
    print(f"   - Context Messages: {len(context_messages)}")
    print(f"   - Translations: {len(translations)}")
    print(f"   - Reference Texts: {len(references)}")
    print(f"   - Model: {model_name}")
    
    # Prepare model data
    models_data = {
        model_name: {
            'outputs': translations,
            'references': references
        }
    }
    
    # Create evaluation engine
    engine = EvaluationEngine(models_data)
    
    # Evaluate
    results = engine.evaluate_all()
    
    # Generate report
    report = engine.generate_report(output_dir)
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)
    print(f"\n📁 Results saved to: {output_dir}")
    
    return results

# ==================== TEST WITH APP DATA ====================

if __name__ == "__main__":
    # Simulate data from app.py
    context_messages = [
        {"sender": "Amit", "text": "Kal milte hai"},
        {"sender": "Neha", "text": "PDF anupu"},
        {"sender": "Arun", "text": "Ooty polama?"},
        {"sender": "Priya", "text": "Nalla irukka?"},
        {"sender": "Karthik", "text": "Meeting timings changed"}
    ]
    
    # Simulate translations from different models
    translation_data = [
        {"sender": "Amit", "original": "Kal milte hai", "translation": "See you tomorrow"},
        {"sender": "Neha", "original": "PDF anupu", "translation": "Send the PDF"},
        {"sender": "Arun", "original": "Ooty polama?", "translation": "Shall we go to Ooty?"},
        {"sender": "Priya", "original": "Nalla irukka?", "translation": "How are you?"},
        {"sender": "Karthik", "original": "Meeting timings changed", "translation": "Meeting timings have changed"}
    ]
    
    # Test with a model name
    results = evaluate_from_app_data(
        translation_data=translation_data,
        context_messages=context_messages,
        model_name="Test_Model",
        output_dir="test_evaluation"
    )
    
    # Print summary of results
    if results and 'metrics' in results:
        print("\n📊 Results Summary:")
        for model_name, model_results in results['metrics'].items():
            metrics = model_results.get('metrics', {})
            print(f"\n  {model_name}:")
            print(f"    BLEU: {metrics.get('bleu', 0):.2f}")
            print(f"    chrF: {metrics.get('chrf', 0):.2f}")
            print(f"    BERTScore: {metrics.get('bertscore', 0):.4f}")
            print(f"    ROUGE-L: {metrics.get('rouge_l', 0):.4f}")
            print(f"    Samples: {metrics.get('n_samples', 0)}")