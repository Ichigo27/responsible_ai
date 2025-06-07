# responsible_ai/utils/usage_logger.py
"""
Centralized usage logging system for tracking LLM and API usage.
"""

import time
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pythonjsonlogger import jsonlogger
import logging
from concurrent_log_handler import ConcurrentTimedRotatingFileHandler
import os


class UsageType(Enum):
    """Types of usage being tracked"""

    LLM_REQUEST = "llm_request"
    OPERATION = "operation"


class OperationType(Enum):
    """Types of LLM operations being tracked"""

    EVALUATE = "evaluate"  # Single prompt-response evaluation
    BATCH_EVALUATE = "batch_evaluate"  # Multiple items evaluation
    EVALUATE_METRIC = "evaluate_metric"  # Single specific metric evaluation


@dataclass
class LLMUsage:
    """Data class for LLM usage metrics"""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    model: str = ""
    provider: str = ""
    latency: float = 0.0

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        """Add two LLMUsage instances together"""
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost=self.cost + other.cost,
            model=self.model or other.model,  # Keep the first non-empty model
            provider=self.provider or other.provider,  # Keep the first non-empty provider
            latency=self.latency + other.latency,
        )


@dataclass
class UsageRecord:
    """Base class for usage records"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    usage_type: UsageType = UsageType.LLM_REQUEST
    status: str = "success"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        data = asdict(self)
        data["usage_type"] = self.usage_type.value
        data["timestamp_iso"] = datetime.fromtimestamp(self.timestamp).isoformat()
        return data


@dataclass
class LLMRequestRecord(UsageRecord):
    """Record for individual LLM requests"""

    usage_type: UsageType = UsageType.LLM_REQUEST
    llm_request_id: str = ""  # Renamed from request_id
    api_request_id: Optional[str] = None  # New field to link to API request
    prompt: str = ""
    response: str = ""
    model: str = ""
    provider: str = ""
    usage: LLMUsage = field(default_factory=LLMUsage)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["usage"] = asdict(self.usage)
        # Truncate long prompts/responses
        data["prompt"] = self.prompt[:500] + "..." if len(self.prompt) > 500 else self.prompt
        data["response"] = self.response[:500] + "..." if len(self.response) > 500 else self.response
        return data


@dataclass
class OperationRecord(UsageRecord):
    """Record for LLM operations"""

    usage_type: UsageType = UsageType.OPERATION
    operation_type: OperationType = OperationType.EVALUATE

    # HTTP context (minimal)
    endpoint: str = ""
    method: str = "POST"
    status_code: int = 200

    # LLM operation context
    api_request_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Renamed from operation_id
    metrics: List[str] = field(default_factory=list)
    item_count: int = 1  # 1 for single evaluate, N for batch

    # LLM usage tracking
    llm_request_count: int = 0  # Number of LLM calls made
    total_time: float = 0.0
    usage: LLMUsage = field(default_factory=LLMUsage)

    # Extensible metadata for future needs
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = super().to_dict()
        data["operation_type"] = self.operation_type.value
        data["usage"] = asdict(self.usage)
        # Only include metadata if it has content
        if self.metadata:
            data["metadata"] = self.metadata
        return data


class UsageLogger:
    """
    Centralized usage logger for tracking LLM and API usage.
    Uses Python JSON Logger for structured JSON logging.
    """

    _instances = {}

    def __new__(cls, logger_name: str = "usage", log_file: str = "usage.log"):
        """Singleton pattern to ensure one logger per name"""
        if logger_name not in cls._instances:
            instance = super(UsageLogger, cls).__new__(cls)
            cls._instances[logger_name] = instance
        return cls._instances[logger_name]

    def __init__(self, logger_name: str = "usage", log_file: str = "usage.log"):
        """Initialize the usage logger"""
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self.logger_name = logger_name
        self.log_file = log_file

        # Load configuration
        try:
            from utils.config_manager import ConfigManager

            self.config = ConfigManager().get_config("app_config")
            self.logs_dir = self.config.get("LOGS_DIR", "logs")
        except:
            self.logs_dir = "logs"

        self._setup_logger()

    def _setup_logger(self):
        """Set up the JSON logger"""
        # Create logger
        self.logger = logging.getLogger(f"usage.{self.logger_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create logs directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_folder = os.path.join(project_root, self.logs_dir, "usage")
        os.makedirs(log_folder, exist_ok=True)

        # Create file handler with JSON formatter
        log_file_path = os.path.join(log_folder, self.log_file)
        file_handler = ConcurrentTimedRotatingFileHandler(filename=log_file_path, when="midnight", backupCount=30)

        # Use JSON formatter
        json_formatter = jsonlogger.JsonFormatter(fmt="%(timestamp)s %(level)s %(name)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(json_formatter)

        self.logger.addHandler(file_handler)

    def log_llm_request(self, record: LLMRequestRecord):
        """Log an LLM request"""
        self.logger.info("llm_request", extra=record.to_dict())

    def log_operation(self, record: OperationRecord):
        """Log an operation"""
        self.logger.info("operation", extra=record.to_dict())

    def log_record(self, record: UsageRecord):
        """Generic method to log any usage record"""
        self.logger.info(record.usage_type.value, extra=record.to_dict())


class UsageTracker:
    """
    Context manager for tracking usage within a scope.
    Automatically logs usage when exiting the context.
    """

    def __init__(self, logger: UsageLogger, record: UsageRecord):
        self.logger = logger
        self.record = record
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Calculate total time
        if hasattr(self.record, "total_time"):
            self.record.total_time = time.time() - self.start_time

        # Set status based on exception
        if exc_type is not None:
            self.record.status = "failed"
            self.record.error_message = str(exc_val)

        # Log the record
        self.logger.log_record(self.record)

    def add_llm_usage(self, usage: LLMUsage):
        """Add LLM usage to the record"""
        if hasattr(self.record, "usage"):
            self.record.usage = self.record.usage + usage
        if hasattr(self.record, "llm_request_count"):
            self.record.llm_request_count += 1

    def update(self, **kwargs):
        """Update record attributes"""
        for key, value in kwargs.items():
            if hasattr(self.record, key):
                setattr(self.record, key, value)


# Create singleton instances for different usage types
def get_llm_usage_logger() -> UsageLogger:
    """Get the LLM usage logger instance"""
    return UsageLogger("llm", "llm_usage.log")


def get_operation_logger() -> UsageLogger:
    """Get the operation logger instance"""
    return UsageLogger("operations", "operations.log")
