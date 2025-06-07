"""
Base evaluator for responsible AI metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from logs import get_logger
from utils.config_manager import ConfigManager


class BaseEvaluator(ABC):
    """
    Abstract base class for evaluators.
    """

    def __init__(self, metric_name: str):
        """
        Initialize the evaluator.

        Args:
            metric_name: Name of the metric this evaluator handles
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config_manager = ConfigManager()
        self.metric_name = metric_name
        self._load_config()

    def _load_config(self):
        """Load metric-specific configuration."""
        metrics_config = self.config_manager.get_config("metrics_prompt_config")
        self.metric_config = metrics_config.get("metrics", {}).get(self.metric_name, {})
        if not self.metric_config:
            self.logger.warning(f"No configuration found for metric: {self.metric_name}")
        else:
            self.logger.info(f"Loaded configuration for metric: {self.metric_name}")
            self.prompt_template = self.metric_config.get("prompt_template", "")
            self.threshold = self.metric_config.get("threshold", 0.7)

    @abstractmethod
    def evaluate(self, prompt: str, response: str, api_request_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Evaluate the response based on the metric.

        Args:
            prompt: The original prompt
            response: The model's response to evaluate
            api_request_id: Optional API request ID for tracing
            **kwargs: Additional parameters for evaluation

        Returns:
            Evaluation result as a dictionary
        """
        pass

    @abstractmethod
    def batch_evaluate(self, data: List[Dict[str, Any]], api_request_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Evaluate multiple prompts and responses.

        Args:
            data: List of dictionaries containing prompts and responses
            api_request_id: Optional API request ID for tracing
            **kwargs: Additional parameters for evaluation

        Returns:
            List of evaluation results
        """
        pass
