# responsible_ai/core/evaluators/rai_evaluator.py
"""
Main evaluator for responsible AI metrics.
"""

import concurrent.futures
import uuid
import time
from typing import Dict, Any, List, Optional
from .base_evaluator import BaseEvaluator
from utils.config_manager import ConfigManager
from utils.errors import EvaluationError
from utils.helpers import preserve_item_context
from core.metrics.bias_fairness.evaluator import BiasFairnessEvaluator
from core.metrics.hallucination.evaluator import HallucinationEvaluator
from core.metrics.toxicity.evaluator import ToxicityEvaluator
from core.metrics.relevance.evaluator import RelevanceEvaluator
from core.metrics.explainability.evaluator import ExplainabilityEvaluator
from logs import get_logger
from utils.usage_logger import LLMUsage, UsageTracker, OperationRecord, OperationType, get_operation_logger


class RAIEvaluator:
    """
    Main evaluator that coordinates evaluation across different metrics.
    """

    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize the RAI evaluator.

        Args:
            metrics: List of metrics to evaluate (default: all available metrics)
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config_manager = ConfigManager()
        self.app_config = self.config_manager.get_config("app_config")

        # Initialize available evaluators
        self.evaluators: Dict[str, BaseEvaluator] = {
            "bias_fairness": BiasFairnessEvaluator(),
            "hallucination": HallucinationEvaluator(),
            "toxicity": ToxicityEvaluator(),
            "relevance": RelevanceEvaluator(),
            "explainability": ExplainabilityEvaluator(),
        }

        # Select which metrics to use
        self.metrics = metrics if metrics else list(self.evaluators.keys())
        self.logger.info(f"RAIEvaluator initialized with metrics: {', '.join(self.metrics)}")

        # Get parallelization configuration
        self.enable_parallel = self.app_config.get("ENABLE_PARALLEL_PROCESSING", True)
        self.max_batch_workers = self.app_config.get("MAX_BATCH_WORKERS", 16)
        self.max_metric_workers = self.app_config.get("MAX_METRIC_WORKERS", 10)

        # Get timeout configuration
        self.metric_result_timeout = self.app_config.get("METRIC_RESULT_TIMEOUT", 300)
        self.batch_item_timeout = self.app_config.get("BATCH_ITEM_TIMEOUT", 600)
        self.batch_chunk_timeout = self.app_config.get("BATCH_CHUNK_TIMEOUT", 1200)

        # Initialize thread pools up front to avoid creating them dynamically
        if self.enable_parallel:
            self.batch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_batch_workers, thread_name_prefix="batch_worker")
            self.metric_executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_metric_workers, thread_name_prefix="metric_worker")
            self.logger.info(f"Parallel processing enabled with batch workers: {self.max_batch_workers}, metric workers: {self.max_metric_workers}")
        else:
            self.batch_executor = None
            self.metric_executor = None
            self.logger.info("Parallel processing is disabled, using sequential processing")

    def evaluate(
        self, item: Dict[str, Any], metrics: Optional[List[str]] = None, tracker: Optional[UsageTracker] = None, api_request_id: Optional[str] = None, **context_params
    ) -> Dict[str, Any]:
        """
        Evaluate a prompt-response pair against selected metrics.

        Args:
            item: Dictionary containing prompt, response, and optional id/metadata
            metrics: Optional list of specific metrics to evaluate
            tracker: Optional UsageTracker from API layer
            api_request_id: Optional API request ID for tracing
            **context_params: Additional context parameters including:
                - system_instructions: System instructions for the LLM
                - conversation_history: Previous conversation turns
                - retrieved_contexts: Retrieved documents or context

        Returns:
            Dictionary with evaluation results for each metric
        """
        evaluation_id = str(uuid.uuid4())
        start_time = time.time()

        prompt = item.get("prompt", "")
        response = item.get("response", "")
        metrics_to_use = metrics if metrics else self.metrics
        self.logger.info(f"Evaluating prompt-response pair with metrics: {', '.join(metrics_to_use)}")
        results = {"prompt": prompt, "response": response, "metrics": {}}

        try:
            # Only track if tracker is provided (from API)
            if tracker:
                tracker.update(api_request_id=api_request_id, metrics=metrics_to_use)

            if self.enable_parallel and len(metrics_to_use) > 1 and self.metric_executor is not None:
                evaluation_result = self._parallel_metric_evaluate(prompt, response, metrics_to_use, results, tracker, api_request_id, **context_params)
            else:
                evaluation_result = self._sequential_metric_evaluate(prompt, response, metrics_to_use, results, tracker, api_request_id, **context_params)

            evaluation_result["evaluation_id"] = evaluation_id

            # Add LLM usage only if tracker exists
            if tracker:
                evaluation_result["llm_usage"] = {
                    "input_tokens": tracker.record.usage.input_tokens,
                    "output_tokens": tracker.record.usage.output_tokens,
                    "total_tokens": tracker.record.usage.total_tokens,
                    "cost": tracker.record.usage.cost,
                    "request_count": tracker.record.llm_request_count,
                }

            return preserve_item_context(item, evaluation_result)

        except Exception as e:
            self.logger.error(f"Error in evaluate: {str(e)}")
            raise

    def _sequential_metric_evaluate(
        self, prompt: str, response: str, metrics_to_use: List[str], results: Dict[str, Any], tracker: Optional[UsageTracker], api_request_id: Optional[str], **context_params
    ) -> Dict[str, Any]:
        """
        Evaluate metrics sequentially for a single prompt-response pair.

        Args:
            prompt: The original prompt
            response: The model's response to evaluate
            metrics_to_use: List of metrics to evaluate
            results: Pre-initialized results dictionary to populate
            tracker: Optional UsageTracker for aggregating usage
            api_request_id: Optional API request ID for tracing
            **context_params: Additional context parameters (system_instructions, conversation_history, retrieved_contexts)

        Returns:
            Dictionary with evaluation results for each metric
        """
        for metric_name in metrics_to_use:
            if metric_name in self.evaluators:
                try:
                    self.logger.info(f"Evaluating metric: {metric_name}")
                    evaluator = self.evaluators[metric_name]
                    metric_result, metric_usage = evaluator.evaluate(prompt, response, api_request_id=api_request_id, **context_params)
                    results["metrics"][metric_name] = metric_result

                    # Add usage to tracker only if it exists
                    if tracker:
                        llm_usage = LLMUsage(
                            input_tokens=metric_usage.get("input_tokens", 0),
                            output_tokens=metric_usage.get("output_tokens", 0),
                            total_tokens=metric_usage.get("total_tokens", 0),
                            cost=metric_usage.get("cost", 0.0),
                            model=metric_usage.get("model", ""),
                            provider=metric_usage.get("provider", ""),
                            latency=metric_usage.get("latency", 0.0),
                        )
                        tracker.add_llm_usage(llm_usage)

                    self.logger.info(f"Metric {metric_name} result: score={metric_result.get('score', 'N/A')}")
                except Exception as e:
                    self.logger.error(f"Error evaluating metric {metric_name}: {str(e)}")
                    results["metrics"][metric_name] = {"error": str(e), "score": 0, "passed": False}
            else:
                self.logger.warning(f"Metric not found: {metric_name}")

        return results

    def _parallel_metric_evaluate(
        self, prompt: str, response: str, metrics_to_use: List[str], results: Dict[str, Any], tracker: Optional[UsageTracker], api_request_id: Optional[str], **context_params
    ) -> Dict[str, Any]:
        """
        Evaluate metrics in parallel for a single prompt-response pair.

        Args:
            prompt: The original prompt
            response: The model's response to evaluate
            metrics_to_use: List of metrics to evaluate
            results: Pre-initialized results dictionary to populate
            tracker: Optional UsageTracker for aggregating usage
            api_request_id: Optional API request ID for tracing
            **context_params: Additional context parameters (system_instructions, conversation_history, retrieved_contexts)

        Returns:
            Dictionary with evaluation results for each metric
        """
        if not self.metric_executor:
            self.logger.warning("Metric executor not initialized, falling back to sequential processing")
            return self._sequential_metric_evaluate(prompt, response, metrics_to_use, results, tracker, api_request_id, **context_params)

        # Safety check - avoid deadlock with too many submissions
        if len(metrics_to_use) > self.max_metric_workers * 2:
            self.logger.warning(f"Too many metrics ({len(metrics_to_use)}) for parallel processing, limiting submission rate")
            # Process in smaller batches to avoid overwhelming the executor
            metrics_batches = [metrics_to_use[i : i + self.max_metric_workers] for i in range(0, len(metrics_to_use), self.max_metric_workers)]
            for batch in metrics_batches:
                self._submit_metric_batch(prompt, response, batch, results, tracker, api_request_id, **context_params)
        else:
            # Submit all metrics at once
            self._submit_metric_batch(prompt, response, metrics_to_use, results, tracker, api_request_id, **context_params)

        return results

    def _handle_metric_future_result(self, future, metric_name, results, tracker: Optional[UsageTracker]):
        try:
            metric_result, metric_usage = future.result(timeout=self.metric_result_timeout)
            results["metrics"][metric_name] = metric_result

            # Add usage to tracker only if it exists
            if tracker:
                llm_usage = LLMUsage(
                    input_tokens=metric_usage.get("input_tokens", 0),
                    output_tokens=metric_usage.get("output_tokens", 0),
                    total_tokens=metric_usage.get("total_tokens", 0),
                    cost=metric_usage.get("cost", 0.0),
                    model=metric_usage.get("model", ""),
                    provider=metric_usage.get("provider", ""),
                    latency=metric_usage.get("latency", 0.0),
                )
                tracker.add_llm_usage(llm_usage)

            self.logger.info(f"Metric {metric_name} result: score={metric_result.get('score', 'N/A')}")
        except concurrent.futures.TimeoutError:
            self.logger.error(f"Timeout while evaluating metric {metric_name}")
            results["metrics"][metric_name] = {"error": "Operation timed out", "score": 0, "passed": False}
        except Exception as e:
            self.logger.error(f"Error evaluating metric {metric_name} in parallel: {str(e)}")
            results["metrics"][metric_name] = {"error": str(e), "score": 0, "passed": False}

    def _submit_metric_task(self, metric_name, prompt, response, futures, results, api_request_id, **context_params):
        """
        Submit a single metric evaluation task to the executor.

        Args:
            metric_name: Name of the metric to evaluate
            prompt: The original prompt
            response: The model's response to evaluate
            futures: Dictionary to store futures
            results: Results dictionary to populate
            api_request_id: Optional API request ID for tracing
            **context_params: Additional context parameters (system_instructions, conversation_history, retrieved_contexts)
        """
        if metric_name in self.evaluators:
            evaluator = self.evaluators[metric_name]
            if self.metric_executor is not None:
                future = self.metric_executor.submit(evaluator.evaluate, prompt, response, api_request_id, **context_params)
                futures[future] = metric_name
            else:
                self.logger.error("Metric executor is not initialized.")
                results["metrics"][metric_name] = {"error": "Metric executor not initialized", "score": 0, "passed": False}
        else:
            self.logger.warning(f"Metric not found: {metric_name}")

    def _handle_pending_metric_errors(self, metrics_batch, results, error):
        for metric_name in metrics_batch:
            if metric_name not in results.get("metrics", {}):
                results["metrics"][metric_name] = {"error": f"Submission error: {str(error)}", "score": 0, "passed": False}

    def _cancel_pending_futures(self, futures):
        for future in futures:
            if not future.done():
                future.cancel()

    def _submit_metric_batch(
        self, prompt: str, response: str, metrics_batch: list, results: dict, tracker: Optional[UsageTracker], api_request_id: Optional[str], **context_params
    ) -> None:
        """
        Submit a batch of metrics for parallel evaluation with proper cleanup.

        Args:
            prompt: The original prompt
            response: The model's response to evaluate
            metrics_batch: List of metrics to evaluate in this batch
            results: Results dictionary to populate
            tracker: Optional UsageTracker for aggregating usage
            api_request_id: Optional API request ID for tracing
            **context_params: Additional context parameters (system_instructions, conversation_history, retrieved_contexts)
        """
        futures = {}
        try:
            for metric_name in metrics_batch:
                self._submit_metric_task(metric_name, prompt, response, futures, results, api_request_id, **context_params)
            for future in concurrent.futures.as_completed(futures, timeout=self.batch_chunk_timeout):
                metric_name = futures[future]
                self._handle_metric_future_result(future, metric_name, results, tracker)
        except Exception as e:
            self.logger.error(f"Error in parallel metric submission: {str(e)}")
            self._handle_pending_metric_errors(metrics_batch, results, e)
        finally:
            self._cancel_pending_futures(futures)

    def batch_evaluate(self, data: List[Dict[str, Any]], metrics: Optional[List[str]] = None, api_request_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Evaluate multiple prompt-response pairs with optional parallel processing.
        Note: This method should be called from API endpoints that handle their own tracking.

        Args:
            data: List of dictionaries, each with prompt and response keys
            metrics: Optional list of specific metrics to evaluate
            api_request_id: Optional API request ID for tracing

        Returns:
            List of evaluation results
        """
        batch_id = str(uuid.uuid4())
        batch_start_time = time.time()

        metrics_to_use = metrics if metrics else self.metrics
        self.logger.info(f"Batch evaluating {len(data)} items with metrics: {', '.join(metrics_to_use)}")

        # Filter out items with missing prompts or responses
        valid_data = []
        for i, item in enumerate(data):
            prompt = item.get("prompt", "")
            response = item.get("response", "")

            if not prompt or not response:
                self.logger.warning(f"Skipping item {i}: missing prompt or response")
                continue

            valid_data.append(item)

        self.logger.info(f"Processing {len(valid_data)} valid items out of {len(data)} total items")

        try:
            # If parallel processing is enabled and we have more than one item, use ThreadPoolExecutor
            if self.enable_parallel and len(valid_data) > 1 and self.batch_executor is not None:
                results = self._parallel_batch_evaluate(valid_data, metrics_to_use, api_request_id)
            else:
                results = self._sequential_batch_evaluate(valid_data, metrics_to_use, api_request_id)

            return results

        except Exception as e:
            self.logger.error(f"Error in batch_evaluate: {str(e)}")
            raise

    def _parallel_batch_evaluate(self, data: List[Dict[str, Any]], metrics_to_use: List[str], api_request_id: Optional[str]) -> List[Dict[str, Any]]:
        """
        Evaluate batch items in parallel using ThreadPoolExecutor.

        Args:
            data: List of valid data items to evaluate
            metrics_to_use: List of metrics to evaluate
            api_request_id: Optional API request ID for tracing

        Returns:
            List of evaluation results
        """
        self.logger.info(f"Starting parallel batch evaluation of {len(data)} items")
        results = []

        # Safety check - avoid deadlock with too many submissions
        if len(data) > self.max_batch_workers * 2:
            self.logger.info(f"Large batch detected ({len(data)} items), processing in smaller chunks")
            # Process in smaller batches to avoid overwhelming the executor
            chunk_size = self.max_batch_workers
            for i in range(0, len(data), chunk_size):
                chunk = data[i : i + chunk_size]
                chunk_results = self._process_batch_chunk(chunk, metrics_to_use, i, api_request_id)
                results.extend(chunk_results)
                self.logger.info(f"Completed chunk {i//chunk_size + 1}/{(len(data) + chunk_size - 1)//chunk_size}")
        else:
            # Process all items at once
            results = self._process_batch_chunk(data, metrics_to_use, 0, api_request_id)

        self.logger.info(f"Parallel batch evaluation complete - processed {len(results)} items")
        return results

    def _handle_batch_future_result(self, future, item, index, chunk_results, offset, data_chunk):
        try:
            result = future.result(timeout=self.batch_item_timeout)
            chunk_results.append(result)
            self.logger.info(f"Completed batch item {index+1}/{offset + len(data_chunk)}")
        except concurrent.futures.TimeoutError:
            self.logger.error(f"Timeout processing batch item {index}")
            error_result = {
                "prompt": item.get("prompt", ""),
                "response": item.get("response", ""),
                "id": item.get("id", f"item-{index}"),
                "error": "Operation timed out",
                "metrics": {},
            }
            chunk_results.append(error_result)
        except Exception as e:
            self.logger.error(f"Error processing batch item {index}: {str(e)}")
            error_result = {
                "prompt": item.get("prompt", ""),
                "response": item.get("response", ""),
                "id": item.get("id", f"item-{index}"),
                "error": str(e),
                "metrics": {},
            }
            chunk_results.append(error_result)

    def _process_batch_chunk(self, data_chunk: list, metrics_to_use: list, offset: int, api_request_id: Optional[str]) -> list:
        """
        Process a chunk of the batch data in parallel.
        """
        chunk_results = []
        futures = {}
        try:
            for i, item in enumerate(data_chunk):
                index = offset + i
                if self.batch_executor is not None:
                    futures[self.batch_executor.submit(self._evaluate_batch_item, item, metrics_to_use, index, api_request_id)] = (item, index)
                else:
                    self.logger.error("Batch executor is not initialized.")
            for future in concurrent.futures.as_completed(list(futures.keys()), timeout=self.batch_chunk_timeout):
                item, index = futures[future]
                self._handle_batch_future_result(future, item, index, chunk_results, offset, data_chunk)
        except Exception as e:
            self.logger.error(f"Error in batch chunk processing: {str(e)}")
        finally:
            self._cancel_pending_futures(futures)
        return chunk_results

    def _sequential_batch_evaluate(self, data: List[Dict[str, Any]], metrics_to_use: List[str], api_request_id: Optional[str]) -> List[Dict[str, Any]]:
        """
        Evaluate batch items sequentially without parallelism.

        Args:
            data: List of valid data items to evaluate
            metrics_to_use: List of metrics to evaluate
            api_request_id: Optional API request ID for tracing

        Returns:
            List of evaluation results
        """
        self.logger.info(f"Starting sequential batch evaluation of {len(data)} items")
        results = []

        for i, item in enumerate(data):
            try:
                result = self._evaluate_batch_item(item, metrics_to_use, i, api_request_id)
                results.append(result)
                self.logger.info(f"Processed batch item {i+1}/{len(data)}")
            except Exception as e:
                self.logger.error(f"Error processing batch item {i}: {str(e)}")
                error_result = {"prompt": item.get("prompt", ""), "response": item.get("response", ""), "id": item.get("id", f"item-{i}"), "error": str(e), "metrics": {}}
                results.append(error_result)

        self.logger.info(f"Sequential batch evaluation complete - processed {len(results)} items")
        return results

    def _evaluate_batch_item(self, item: Dict[str, Any], metrics_to_use: List[str], index: int, api_request_id: Optional[str]) -> Dict[str, Any]:
        """
        Evaluate a single batch item.

        Args:
            item: Data item to evaluate
            metrics_to_use: List of metrics to evaluate
            index: Item index for logging
            api_request_id: Optional API request ID for tracing

        Returns:
            Evaluation result for the item
        """
        prompt = item.get("prompt", "")
        response = item.get("response", "")

        try:
            self.logger.info(f"Evaluating batch item {index}")

            # Extract context parameters from the item if available
            context_params = {
                "system_instructions": item.get("system_instructions"),
                "conversation_history": item.get("conversation_history"),
                "retrieved_contexts": item.get("retrieved_contexts"),
            }

            # Create a temporary operation record just for tracking usage
            temp_record = OperationRecord(api_request_id=api_request_id, operation_type=OperationType.EVALUATE, metrics=metrics_to_use)

            # Create a no-op logger that doesn't actually log
            class NoOpLogger:
                def log_record(self, record):
                    pass  # Don't actually log

            no_op_logger = NoOpLogger()

            # Create a temporary tracker to collect usage without logging
            with UsageTracker(no_op_logger, temp_record) as temp_tracker:
                # Call evaluate with the temporary tracker to collect usage
                result = self.evaluate(item, metrics_to_use, tracker=temp_tracker, api_request_id=api_request_id, **context_params)

            # The result now includes llm_usage from the tracker
            return result

        except Exception as e:
            self.logger.error(f"Error evaluating batch item {index}: {str(e)}")
            raise EvaluationError(f"Failed to evaluate item {index}: {str(e)}")

    def __del__(self):
        """Clean up thread pools when the evaluator is destroyed."""
        try:
            if hasattr(self, "batch_executor") and self.batch_executor:
                self.batch_executor.shutdown(wait=False)
            if hasattr(self, "metric_executor") and self.metric_executor:
                self.metric_executor.shutdown(wait=False)
            self.logger.debug("Thread pools have been shut down")
        except Exception as e:
            # In case of errors during shutdown, just log and continue
            if hasattr(self, "logger"):
                self.logger.error(f"Error shutting down thread pools: {str(e)}")