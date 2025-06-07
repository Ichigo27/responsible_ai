"""
API endpoint definitions.
"""
from flask import request, jsonify
from core.evaluators.rai_evaluator import RAIEvaluator
from core.processors.jsonl_processor import JSONLProcessor
from utils.schema_validator import validate_evaluation_request, validate_batch_request
from utils.errors import RAIBaseError, ValidationError
from logs import get_logger
from utils.usage_logger import get_operation_logger, OperationRecord, OperationType, UsageTracker
import uuid

logger = get_logger(__name__)
jsonl_processor = JSONLProcessor()
rai_evaluator = RAIEvaluator()


def register_api_endpoints(blueprint):
    """
    Register API endpoints with the provided blueprint.

    Args:
        blueprint: Flask Blueprint to register endpoints with
    """

    @blueprint.route("/evaluate", methods=["POST"])
    def evaluate():
        """
        Evaluate a single prompt-response pair.
        """
        operation_logger = get_operation_logger()

        # Create unique API request ID
        api_request_id = str(uuid.uuid4())

        # Create operation record with api_request_id
        operation = OperationRecord(
            operation_type=OperationType.EVALUATE, endpoint="/api/v1/evaluate", method=request.method, api_request_id=api_request_id  # Using the new field name
        )

        with UsageTracker(operation_logger, operation) as tracker:
            try:
                data = request.json
                validate_evaluation_request(data)

                metrics = data.get("metrics", None)  # Optional list of metrics to evaluate

                # Extract optional context parameters
                context_params = {
                    "system_instructions": data.get("system_instructions"),
                    "conversation_history": data.get("conversation_history"),
                    "retrieved_contexts": data.get("retrieved_contexts"),
                }

                # Update tracker with metrics info
                tracker.update(metrics=metrics or [])

                # Pass tracker, context, and api_request_id to evaluator
                result = rai_evaluator.evaluate(data, metrics=metrics, tracker=tracker, api_request_id=api_request_id, **context_params)  # Pass the api_request_id

                tracker.update(status_code=200)
                return jsonify(result)

            except ValidationError as e:
                logger.error(f"Validation error: {str(e)}")
                tracker.update(status_code=400)
                return jsonify({"error": str(e)}), 400

            except RAIBaseError as e:
                logger.error(f"Evaluation error: {str(e)}")
                tracker.update(status_code=500)
                return jsonify({"error": str(e)}), 500

            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                tracker.update(status_code=500)
                return jsonify({"error": "Internal server error"}), 500

    @blueprint.route("/batch-evaluate", methods=["POST"])
    def batch_evaluate():
        """
        Evaluate multiple prompt-response pairs.
        """
        operation_logger = get_operation_logger()

        # Create unique API request ID
        api_request_id = str(uuid.uuid4())

        # Create operation record with api_request_id
        operation = OperationRecord(
            operation_type=OperationType.BATCH_EVALUATE, endpoint="/api/v1/batch-evaluate", method=request.method, api_request_id=api_request_id  # Using the new field name
        )

        with UsageTracker(operation_logger, operation) as tracker:
            try:
                # Check if request is JSONL format or JSON list
                content_type = request.headers.get("Content-Type", "")

                if "application/jsonl" in content_type:
                    # Process as JSONL
                    data = jsonl_processor.load_data(request.data.decode("utf-8"))
                else:
                    # Process as JSON list
                    data = request.json
                    validate_batch_request(data)

                # Get optional parameters
                metrics_param = request.args.get("metrics")
                metrics = metrics_param.split(",") if metrics_param else None

                # Update tracker with request info
                tracker.update(metrics=metrics or [], item_count=len(data))

                # Evaluate all items - pass api_request_id
                results = rai_evaluator.batch_evaluate(data, metrics=metrics, api_request_id=api_request_id)  # Pass the api_request_id

                # Update tracker with usage from results
                total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost": 0.0, "request_count": 0}

                for result in results:
                    if "llm_usage" in result:
                        usage = result["llm_usage"]
                        total_usage["input_tokens"] += usage.get("input_tokens", 0)
                        total_usage["output_tokens"] += usage.get("output_tokens", 0)
                        total_usage["total_tokens"] += usage.get("total_tokens", 0)
                        total_usage["cost"] += usage.get("cost", 0.0)
                        total_usage["request_count"] += usage.get("request_count", 0)

                # Update tracker with aggregated usage
                tracker.record.usage.input_tokens = total_usage["input_tokens"]
                tracker.record.usage.output_tokens = total_usage["output_tokens"]
                tracker.record.usage.total_tokens = total_usage["total_tokens"]
                tracker.record.usage.cost = total_usage["cost"]
                tracker.record.llm_request_count = total_usage["request_count"]

                # Return JSONL format
                jsonl_response = jsonl_processor.save_results(results)

                tracker.update(status_code=200)
                return jsonl_response, 200, {"Content-Type": "application/jsonl"}

            except ValidationError as e:
                logger.error(f"Validation error: {str(e)}")
                tracker.update(status_code=400)
                return jsonify({"error": str(e)}), 400

            except RAIBaseError as e:
                logger.error(f"Evaluation error: {str(e)}")
                tracker.update(status_code=500)
                return jsonify({"error": str(e)}), 500

            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                tracker.update(status_code=500)
                return jsonify({"error": "Internal server error"}), 500

    @blueprint.route("/metrics/<metric_name>", methods=["POST"])
    def evaluate_specific_metric(metric_name):
        """
        Evaluate a specific metric for a prompt-response pair.
        """
        operation_logger = get_operation_logger()

        # Create unique API request ID
        api_request_id = str(uuid.uuid4())

        # Create operation record with api_request_id
        operation = OperationRecord(
            operation_type=OperationType.EVALUATE_METRIC,
            endpoint=f"/api/v1/metrics/{metric_name}",
            method=request.method,
            metrics=[metric_name],
            api_request_id=api_request_id,  # Using the new field name
        )

        with UsageTracker(operation_logger, operation) as tracker:
            try:
                data = request.json
                validate_evaluation_request(data)

                if metric_name not in rai_evaluator.evaluators:
                    tracker.update(status_code=404)
                    return jsonify({"error": f"Metric not found: {metric_name}"}), 404

                # Extract optional context parameters
                context_params = {
                    "system_instructions": data.get("system_instructions"),
                    "conversation_history": data.get("conversation_history"),
                    "retrieved_contexts": data.get("retrieved_contexts"),
                }

                evaluator = rai_evaluator.evaluators[metric_name]
                result, usage = evaluator.evaluate(data["prompt"], data["response"], api_request_id=api_request_id, **context_params)  # Pass the api_request_id

                # Include ID if provided in request
                if "id" in data:
                    result["id"] = data["id"]

                # Add usage to tracker
                tracker.record.usage.input_tokens = usage.get("input_tokens", 0)
                tracker.record.usage.output_tokens = usage.get("output_tokens", 0)
                tracker.record.usage.total_tokens = usage.get("total_tokens", 0)
                tracker.record.usage.cost = usage.get("cost", 0.0)
                tracker.record.llm_request_count = 1

                # Add metric-specific metadata
                tracker.update(
                    metadata={"metric_name": metric_name, "score": result.get("score", 0), "passed": result.get("passed", False), "threshold": result.get("threshold", 0)}
                )

                tracker.update(status_code=200)
                return jsonify(result)

            except ValidationError as e:
                logger.error(f"Validation error: {str(e)}")
                tracker.update(status_code=400)
                return jsonify({"error": str(e)}), 400

            except RAIBaseError as e:
                logger.error(f"Evaluation error: {str(e)}")
                tracker.update(status_code=500)
                return jsonify({"error": str(e)}), 500

            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                tracker.update(status_code=500)
                return jsonify({"error": "Internal server error"}), 500
