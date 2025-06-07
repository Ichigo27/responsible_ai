"""
Helper functions for the Responsible AI module.
"""

import json
import re
import concurrent.futures
import functools
from typing import Dict, Any, Optional, Callable, Type, TypeVar
from json_repair import repair_json
from logs import get_logger

logger = get_logger(__name__)

# TypeVar for the return type of the decorated function
R = TypeVar("R")


def thread_timeout(timeout_exception: Type[Exception], timeout: Optional[int] = 300):
    """
    Thread-safe timeout decorator for any operation.

    Args:
        timeout_exception: Exception to raise on timeout
        timeout: Timeout value in seconds (default 300)

    Returns:
        Decorated function with timeout enforcement
    """

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> R:
            # Store the timeout value on the wrapper for dynamic updates
            if not hasattr(wrapper, "__timeout__"):
                wrapper.__timeout__ = timeout

            # Use the current timeout setting (allows for dynamic changes)
            current_timeout = wrapper.__timeout__

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, self, *args, **kwargs)
                try:
                    return future.result(timeout=current_timeout)
                except concurrent.futures.TimeoutError:
                    logger.error(f"Operation '{func.__name__}' timed out after {current_timeout} seconds")
                    raise timeout_exception(f"Operation timed out after {current_timeout} seconds")

        return wrapper

    return decorator


def parse_json_from_string(text: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from a string that might contain additional text.

    Args:
        text: String that may contain JSON

    Returns:
        Parsed JSON as a dictionary
    """
    try:
        # Try to find JSON using regex for more robust extraction
        json_pattern = r"(\{.*\})"
        match = re.search(json_pattern, text, re.DOTALL)

        if match:
            json_str = match.group(1)
            good_json_string = repair_json(json_str)
            return json.loads(good_json_string)

        # Fallback: Look for the first '{' and the last '}'
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:
            json_str = text[start : end + 1]
            good_json_string = repair_json(json_str)
            return json.loads(good_json_string)

        # If we get here, no JSON was found
        raise ValueError("No JSON object found in the string")

    except Exception as e:
        raise ValueError(f"Error parsing and repairing JSON: {e}")


def format_metric_result(score: float, reason: str, threshold: float, additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Format metric evaluation result into a standardized structure.

    Args:
        score: Numeric score from the evaluation
        reason: Explanation for the score
        threshold: Threshold for passing/failing this metric
        additional_data: Any additional metric-specific data

    Returns:
        Formatted result dictionary
    """
    result = {
        "score": score,
        "reason": reason,
        "threshold": threshold,
        "passed": score >= threshold,
    }

    if additional_data:
        result["additional_data"] = additional_data

    return result


def preserve_item_context(source_item: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preserve ID and metadata from source item to result object.

    Args:
        source_item: Original item containing context fields
        result: Evaluation result to enhance with context

    Returns:
        Result with preserved ID and metadata
    """
    if "id" in source_item:
        result["id"] = source_item["id"]
    if "metadata" in source_item:
        result["metadata"] = source_item["metadata"]
    return result
