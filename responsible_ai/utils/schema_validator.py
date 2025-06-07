"""
Schema validation for request and response data.
"""

import jsonschema
from typing import Dict, Any, List, cast
from utils.errors import ValidationError
from logs import get_logger

logger = get_logger(__name__)

# Define schemas with proper typing for Python 3.13
REQUEST_SCHEMA = {
    "type": "object",
    "required": ["prompt", "response"],
    "properties": {
        "prompt": {"type": "string"},
        "response": {"type": "string"},
        "id": {"type": "string"},
        "metadata": {"type": "object"},
        # New optional context fields
        "system_instructions": {"type": "string"},
        "conversation_history": {"type": "string"},
        "retrieved_contexts": {"type": "string"},
    },
    "additionalProperties": True,  # Allow additional properties
}

BATCH_REQUEST_SCHEMA = {"type": "array", "items": REQUEST_SCHEMA}


def validate_evaluation_request(data: Dict[str, Any]) -> None:
    """
    Validate a single evaluation request.

    Args:
        data: Request data to validate

    Raises:
        ValidationError: If validation fails
    """
    try:
        jsonschema.validate(instance=data, schema=cast(Dict[str, Any], REQUEST_SCHEMA))
    except jsonschema.exceptions.ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise ValidationError(f"Invalid request data: {str(e)}")


def validate_batch_request(data: List[Dict[str, Any]]) -> None:
    """
    Validate a batch evaluation request.

    Args:
        data: List of request data to validate

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(data, list):
        raise ValidationError("Batch request must be a list of objects")

    try:
        jsonschema.validate(instance=data, schema=cast(Dict[str, Any], BATCH_REQUEST_SCHEMA))
    except jsonschema.exceptions.ValidationError as e:
        logger.error(f"Batch validation error: {str(e)}")
        raise ValidationError(f"Invalid batch request data: {str(e)}")
