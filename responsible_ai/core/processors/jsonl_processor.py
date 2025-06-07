"""
Processor for handling JSONL data.
"""

import json
from typing import Dict, Any, List, Optional, Union, IO
from .base_processor import BaseProcessor
from utils.errors import DataProcessingError


class JSONLProcessor(BaseProcessor):
    """
    Processor for JSONL formatted data.
    Handles loading, parsing, and saving JSONL files or data.
    """

    def __init__(self):
        """Initialize the JSONL processor."""
        super().__init__()

    def load_data(self, data_source: Union[str, List[str], IO]) -> List[Dict[str, Any]]:
        """
        Load data from JSONL source.

        Args:
            data_source: Source of data - can be file path, list of JSON strings, or file-like object

        Returns:
            List of dictionaries representing the data
        """
        try:
            result = []

            # Handle string (file path)
            if isinstance(data_source, str) and data_source.endswith(".jsonl"):
                self.logger.info(f"Loading JSONL from file: {data_source}")
                with open(data_source, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            result.append(json.loads(line))

            # Handle file-like object
            elif hasattr(data_source, "read") and hasattr(data_source, "readline"):
                self.logger.info("Loading JSONL from file-like object")
                for line in data_source:
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    if line.strip():
                        result.append(json.loads(line))

            # Handle list of JSON strings
            elif isinstance(data_source, list):
                self.logger.info(f"Processing {len(data_source)} JSON strings")
                for item in data_source:
                    if isinstance(item, str) and item.strip():
                        result.append(json.loads(item))
                    elif isinstance(item, dict):
                        result.append(item)

            # Handle a single JSON string
            elif isinstance(data_source, str):
                self.logger.info("Processing a single JSON string")
                result = [json.loads(line) for line in data_source.splitlines() if line.strip()]

            else:
                raise DataProcessingError(f"Unsupported data source type: {type(data_source)}")

            self.logger.info(f"Successfully loaded {len(result)} records")
            return result

        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing error: {str(e)}")
            raise DataProcessingError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error loading JSONL data: {str(e)}")
            raise DataProcessingError(f"Error loading data: {str(e)}")

    def process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process the loaded JSONL data.
        This base implementation just passes through the data.
        Override in subclasses for specific processing.

        Args:
            data: Data to process

        Returns:
            Processed data
        """
        return data

    def save_results(self, results: List[Dict[str, Any]], destination: Optional[Union[str, IO]] = None) -> Any:
        """
        Save results in JSONL format.

        Args:
            results: Results to save
            destination: Optional file path or file-like object

        Returns:
            JSONL string if no destination provided, otherwise None
        """
        try:
            # Convert results to JSONL format
            jsonl_output = "\n".join(json.dumps(item, ensure_ascii=False) for item in results)

            # If destination is a file path, write to file
            if isinstance(destination, str):
                self.logger.info(f"Saving results to file: {destination}")
                with open(destination, "w", encoding="utf-8") as f:
                    f.write(jsonl_output)
                return destination

            # If destination is a file-like object, write to it
            elif hasattr(destination, "write"):
                self.logger.info("Saving results to file-like object")
                destination.write(jsonl_output)
                return destination

            # If no destination, return the JSONL string
            else:
                self.logger.info("Returning results as JSONL string")
                return jsonl_output

        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise DataProcessingError(f"Error saving results: {str(e)}")
