"""
LLM Client implementation using LiteLLM.
"""

import litellm
from litellm import completion_cost
import threading
import time
import random
import uuid
from threading import BoundedSemaphore
from typing import Dict, Any, Optional, Tuple
from utils.config_manager import ConfigManager
from utils.errors import LLMError
from logs import get_logger
from utils.helpers import thread_timeout
from utils.usage_logger import get_llm_usage_logger, LLMRequestRecord, LLMUsage, UsageTracker


class LLMClient:
    """
    Client for interacting with LLMs through LiteLLM.
    Supports multiple models, fallback mechanisms, retries and concurrency control.
    """

    # Class-level semaphore and lock for thread safety
    _llm_semaphore = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize the LLM client with configuration."""
        self.logger = get_logger(__name__)
        self.config_manager = ConfigManager()
        self.llm_config = self.config_manager.get_config("llm_config")
        self.app_config = self.config_manager.get_config("app_config")

        # Set default model
        self.default_model = self.llm_config.get("default_model")
        self.fallback_model = self.llm_config.get("fallback_model")

        # Set up provider configurations
        self._provider_settings = {}
        self._configure_providers()

        # Initialize the semaphore in a thread-safe manner if it doesn't exist
        with LLMClient._lock:
            if LLMClient._llm_semaphore is None:
                # Get concurrency limit from config, with a default of 5
                max_concurrent = self.app_config.get("LLM_MAX_CONCURRENT_REQUESTS", 5)
                LLMClient._llm_semaphore = BoundedSemaphore(value=max_concurrent)
                self.logger.info(f"Initialized LLM semaphore with {max_concurrent} concurrent requests limit")

        self.usage_logger = get_llm_usage_logger()
        self.logger.info(f"LLMClient initialized with default model: {self.default_model}")

    def _configure_providers(self) -> None:
        """Configure LiteLLM providers from config."""
        providers = self.llm_config.get("providers", {})

        # Store provider settings for later use
        for provider, settings in providers.items():
            if provider == "bedrock":
                profile_name = settings.get("profile_name")
                if profile_name:
                    # Update for newer versions of litellm that might have different API
                    try:
                        litellm.aws_profile_name = profile_name
                    except AttributeError:
                        # If attribute error, try the newer method if available
                        litellm.set_aws_profile(profile_name)
                    self.logger.info(f"Configured AWS profile for Bedrock: {profile_name}")

                # Store region setting if provided
                if "aws_region_name" in settings:
                    self._provider_settings["aws_region_name"] = settings["aws_region_name"]
                    self.logger.info(f"Configured AWS region for Bedrock: {settings['aws_region_name']}")

    def _calculate_retry_delay(self, attempt: int, base_delay: float, max_delay: float, backoff_factor: float, jitter: float = 0.1) -> float:
        """
        Calculate delay for retry with exponential backoff and jitter.

        Args:
            attempt: Current attempt number (1-based)
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            backoff_factor: Exponential backoff factor
            jitter: Jitter factor (0-1) to add randomness

        Returns:
            Delay in seconds
        """
        # Calculate exponential backoff
        delay = min(max_delay, base_delay * (backoff_factor ** (attempt - 1)))

        # Add jitter
        jitter_amount = delay * jitter
        delay = delay + random.uniform(-jitter_amount, jitter_amount)

        return max(0.1, delay)  # Ensure minimum delay of 0.1s

    class _SemaphoreAcquire:
        def __init__(self, semaphore, timeout, logger, model_name):
            self.semaphore = semaphore
            self.timeout = timeout
            self.logger = logger
            self.model_name = model_name
            self.acquired = False

        def __enter__(self):
            self.logger.debug(f"Attempting to acquire LLM semaphore for model {self.model_name}")
            self.acquired = self.semaphore.acquire(timeout=self.timeout)
            if not self.acquired:
                self.logger.error(f"Failed to acquire LLM semaphore within {self.timeout}s timeout")
                raise LLMError(f"Too many concurrent LLM requests. Failed to acquire semaphore within {self.timeout}s timeout.")
            self.logger.debug(f"Acquired LLM semaphore for model {self.model_name}")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.acquired:
                self.semaphore.release()
                self.logger.debug(f"Released LLM semaphore for model {self.model_name}")

    def _get_provider_from_model(self, model_str: str) -> str:
        """Extract provider from model string"""
        if "/" in model_str:
            return model_str.split("/")[0]
        return "unknown"

    def _try_fallback_model(self, prompt, kwargs, max_retries, retry_min_delay, retry_backoff_factor, retry_max_delay, model_name, llm_request_id, api_request_id):
        if self.fallback_model and self.fallback_model != model_name:
            self.logger.info(f"Attempting fallback to model: {self.fallback_model}")
            try:
                fallback_config = self._get_model_config(self.fallback_model)
                fallback_params = {**self._provider_settings, **fallback_config, **kwargs}
                fallback_model_str = fallback_params.pop("model", self.fallback_model)
                fallback_response = self._call_with_retries(fallback_model_str, prompt, fallback_params, max_retries, retry_min_delay, retry_backoff_factor, retry_max_delay)
                return fallback_response.choices[0].message.content or "", {
                    "request_id": f"{llm_request_id}-fallback",  # Keep for backward compatibility
                    "model": fallback_model_str,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0,
                    "latency": 0.0,
                    "status": "success",
                }
            except Exception as fallback_error:
                self.logger.error(f"Fallback model also failed: {str(fallback_error)}")
                raise LLMError(f"Both primary and fallback models failed: {str(fallback_error)}")
        else:
            raise LLMError(f"No fallback model available or fallback model is the same as the primary model: {model_name}")

    def get_completion(self, prompt: str, model: Optional[str] = None, api_request_id: Optional[str] = None, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Get a completion from the LLM with retry logic and concurrency control.

        Args:
            prompt: The prompt to send to the LLM
            model: Optional model override
            api_request_id: Optional API request ID for tracing
            **kwargs: Additional parameters to pass to the LLM

        Returns:
            Tuple of (response_text, usage_data)
        """
        model_name = model or self.default_model
        model_config = self._get_model_config(model_name)

        llm_request_id = str(uuid.uuid4())
        start_time = time.time()

        # Get retry configuration from app config
        max_retries = self.app_config.get("MAX_RETRIES", 5)
        retry_min_delay = self.app_config.get("RETRY_MIN_DELAY", 2)
        retry_backoff_factor = self.app_config.get("RETRY_BACKOFF_FACTOR", 2.0)
        retry_max_delay = self.app_config.get("RETRY_MAX_DELAY", 120)
        semaphore_timeout = self.app_config.get("LLM_SEMAPHORE_TIMEOUT", 600)

        # Merge model config with provider settings and custom overrides
        params = {**self._provider_settings, **model_config, **kwargs}

        # Extract the model string
        model_str = params.pop("model", model_name)
        provider = self._get_provider_from_model(model_str)

        # Create LLM request record with new field names
        llm_record = LLMRequestRecord(
            llm_request_id=llm_request_id,  # New field name
            api_request_id=api_request_id,  # New field to link to API request
            prompt=prompt,
            model=model_str,
            provider=provider,
        )

        try:
            with self._SemaphoreAcquire(LLMClient._llm_semaphore, semaphore_timeout, self.logger, model_name):
                self.logger.info(f"Sending request to LLM model: {model_str}")

                with UsageTracker(self.usage_logger, llm_record) as tracker:
                    try:
                        response = self._call_with_retries(model_str, prompt, params, max_retries, retry_min_delay, retry_backoff_factor, retry_max_delay)
                        content = response.choices[0].message.content

                        # Extract usage data
                        usage = LLMUsage()
                        if hasattr(response, "usage"):
                            usage.input_tokens = getattr(response.usage, "prompt_tokens", 0)
                            usage.output_tokens = getattr(response.usage, "completion_tokens", 0)
                            usage.total_tokens = getattr(response.usage, "total_tokens", 0)

                            # Calculate cost
                            try:
                                usage.cost = completion_cost(completion_response=response, model=model_str)
                            except Exception as e:
                                self.logger.warning(f"Error calculating cost: {str(e)}")

                        usage.model = model_str
                        usage.provider = provider
                        usage.latency = time.time() - start_time

                        # Update tracker
                        tracker.record.response = content
                        tracker.record.usage = usage

                        # Return in the expected format
                        usage_data = {
                            "request_id": llm_request_id,  # Keep for backward compatibility
                            "model": model_str,
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "total_tokens": usage.total_tokens,
                            "cost": usage.cost,
                            "latency": usage.latency,
                            "status": "success",
                        }

                        return content, usage_data

                    except Exception as e:
                        self.logger.error(f"Error with primary model {model_name}: {str(e)}")
                        # The UsageTracker will automatically log the failure
                        return self._try_fallback_model(
                            prompt, kwargs, max_retries, retry_min_delay, retry_backoff_factor, retry_max_delay, model_name, llm_request_id, api_request_id
                        )

        except LLMError as err:
            raise LLMError(f"LLM request failed: {str(err)}")

    def _handle_retry_exception(self, e, attempts, max_retries, model, last_error):
        last_error = str(e)
        error_type = type(e).__name__
        if attempts < max_retries:
            self.logger.warning(f"Error with model {model} (attempt {attempts}/{max_retries}): {error_type}: {last_error}")
            delay = self._calculate_retry_delay(
                attempt=attempts,
                base_delay=self.app_config.get("RETRY_MIN_DELAY", 2),
                max_delay=self.app_config.get("RETRY_MAX_DELAY", 120),
                backoff_factor=self.app_config.get("RETRY_BACKOFF_FACTOR", 2.0),
            )
            self.logger.info(f"Waiting {delay:.2f}s before retry {attempts+1}/{max_retries}")
            time.sleep(delay)
        else:
            self.logger.error(f"Error with model {model} (final attempt {attempts}/{max_retries}): {error_type}: {last_error}")
        return last_error

    @thread_timeout(LLMError, timeout=None)
    def _call_with_retries(
        self, model: str, prompt: str, params: Dict[str, Any], max_retries: int, retry_min_delay: float, retry_backoff_factor: float, retry_max_delay: float
    ) -> Any:
        """
        Call the LLM API with retry logic.
        This method is protected by a thread-safe timeout decorator.

        Args:
            model: Model identifier
            prompt: The prompt to send
            params: Additional parameters for the request
            max_retries: Maximum number of retries
            retry_min_delay: Initial delay between retries
            retry_backoff_factor: Factor for exponential backoff
            retry_max_delay: Maximum delay between retries

        Returns:
            Model response

        Raises:
            LLMError: If all retries fail
        """
        attempts = 0
        last_error = None

        # Set timeout for the entire operation
        operation_timeout = self.app_config.get("LLM_OPERATION_TIMEOUT", 600)
        # Update the decorator's timeout dynamically
        self._call_with_retries.__wrapped__.__timeout__ = operation_timeout

        start_time = time.time()

        while attempts < max_retries:
            attempts += 1
            try:
                # Log attempt number if it's a retry
                if attempts > 1:
                    self.logger.info(f"Retry attempt {attempts}/{max_retries} for model {model}")

                # Make the API call
                response = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}], **params)

                content = response.choices[0].message.content

                # Log success with attempt number if it was a retry
                if attempts > 1:
                    elapsed = time.time() - start_time
                    self.logger.info(f"Request succeeded on attempt {attempts}/{max_retries} after {elapsed:.2f}s")

                return response

            except Exception as e:
                last_error = self._handle_retry_exception(e, attempts, max_retries, model, last_error)

        # All retries failed
        elapsed_time = time.time() - start_time
        error_msg = f"All {max_retries} retry attempts failed over {elapsed_time:.2f}s: {last_error}"
        self.logger.error(error_msg)
        raise LLMError(error_msg)

    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dict containing model configuration
        """
        # Get model-specific configurations
        models_config = self.llm_config.get("models", {})
        return models_config.get(model_name, {})
