"""
Package for responsible AI metrics.
"""

from core.metrics.base_metric import BaseMetric

# Import all metrics for easy access
from core.metrics.bias_fairness.evaluator import BiasFairnessEvaluator
from core.metrics.hallucination.evaluator import HallucinationEvaluator
from core.metrics.toxicity.evaluator import ToxicityEvaluator
from core.metrics.relevance.evaluator import RelevanceEvaluator
from core.metrics.explainability.evaluator import ExplainabilityEvaluator

# Registry of all available metrics and their evaluators
AVAILABLE_METRICS = {
    "bias_fairness": BiasFairnessEvaluator,
    "hallucination": HallucinationEvaluator,
    "toxicity": ToxicityEvaluator,
    "relevance": RelevanceEvaluator,
    "explainability": ExplainabilityEvaluator,
}

__all__ = ["BaseMetric", "BiasFairnessEvaluator", "HallucinationEvaluator", "ToxicityEvaluator", "RelevanceEvaluator", "ExplainabilityEvaluator", "AVAILABLE_METRICS"]
