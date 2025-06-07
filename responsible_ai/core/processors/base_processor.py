"""
Base processor for handling data input/output.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from logs import get_logger


class BaseProcessor(ABC):
    """
    Abstract base class for data processors.
    """

    def __init__(self):
        """Initialize the processor."""
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def load_data(self, data_source: Any) -> List[Dict[str, Any]]:
        """
        Load data from source.

        Args:
            data_source: Source of data (file path, request data, etc.)

        Returns:
            List of dictionaries representing the data
        """
        pass

    @abstractmethod
    def process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process the loaded data.

        Args:
            data: Data to process

        Returns:
            Processed data
        """
        pass

    @abstractmethod
    def save_results(self, results: List[Dict[str, Any]], destination: Optional[Any] = None) -> Any:
        """
        Save processing results.

        Args:
            results: Results to save
            destination: Optional destination

        Returns:
            Information about the save operation
        """
        pass
