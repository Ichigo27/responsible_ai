"""
Custom error classes for the Responsible AI module.
"""


class RAIBaseError(Exception):
    """Base error class for all Responsible AI errors."""

    pass


class ConfigurationError(RAIBaseError):
    """Error raised for configuration issues."""

    pass


class DataProcessingError(RAIBaseError):
    """Error raised for data processing issues."""

    pass


class EvaluationError(RAIBaseError):
    """Error raised for evaluation issues."""

    pass


class LLMError(RAIBaseError):
    """Error raised for LLM-related issues."""

    pass


class ValidationError(RAIBaseError):
    """Error raised for data validation issues."""

    pass
