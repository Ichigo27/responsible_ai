"""
Base class for all metric implementations.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from utils.config_manager import ConfigManager
from logs import get_logger

class BaseMetric(ABC):
    """
    Abstract base class for all metrics.
    Defines the interface that all metrics must implement.
    """
    
    def __init__(self, metric_name: str):
        """
        Initialize the metric.
        
        Args:
            metric_name: Name of the metric
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config_manager = ConfigManager()
        self.metric_name = metric_name
        self._load_config()
    
    def _load_config(self):
        """
        Load metric-specific configuration from metrics_prompt_config.yaml.
        """
        metrics_config = self.config_manager.get_config("metrics_prompt_config")
        self.metric_config = metrics_config.get("metrics", {}).get(self.metric_name, {})
        
        if not self.metric_config:
            self.logger.warning(f"No configuration found for metric: {self.metric_name}")
        else:
            self.logger.info(f"Loaded configuration for metric: {self.metric_name}")
            self.name = self.metric_config.get("name", self.metric_name)
            self.type = self.metric_config.get("type", "score")
            self.prompt_template = self.metric_config.get("prompt_template", "")
            self.threshold = self.metric_config.get("threshold", 0.7)
    
    @abstractmethod
    def calculate(self, prompt: str, response: str, **kwargs) -> Dict[str, Any]:
        """
        Calculate the metric for a prompt-response pair.
        
        Args:
            prompt: Original prompt
            response: Model's response to evaluate
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with metric results
        """
        pass
    
    @abstractmethod
    def batch_calculate(self, data: List[Dict[str, str]], **kwargs) -> List[Dict[str, Any]]:
        """
        Calculate the metric for multiple prompt-response pairs.
        
        Args:
            data: List of dictionaries with prompt-response pairs
            **kwargs: Additional parameters
            
        Returns:
            List of metric results
        """
        pass
    
    @abstractmethod
    def get_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary from multiple metric calculations.
        
        Args:
            results: List of metric calculation results
            
        Returns:
            Summary statistics
        """
        pass
    
    def __str__(self) -> str:
        """String representation of the metric."""
        return f"{self.name} Metric"
    
    def __repr__(self) -> str:
        """Representation of the metric."""
        return f"<{self.__class__.__name__}(name={self.name}, type={self.type})>"
