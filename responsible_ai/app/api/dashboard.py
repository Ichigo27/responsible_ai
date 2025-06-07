"""
API endpoints for managing dashboard datasets.
"""

from flask import Blueprint, request, jsonify, current_app
from core.processors.jsonl_processor import JSONLProcessor
from utils.errors import ValidationError
from utils.schema_validator import validate_batch_request
from logs import get_logger
import os
import json

dashboard_bp = Blueprint("dashboard", __name__, url_prefix="/dashboard")
logger = get_logger(__name__)
jsonl_processor = JSONLProcessor()


def get_dataset_path():
    """
    Get the configured dashboard dataset path and ensure it exists.

    Returns:
        str: Path to the dashboard dataset directory
    """
    # Get dashboard config from app config
    dashboard_config = current_app.config.get("DASHBOARD", {})
    dataset_path = dashboard_config.get("DATASET_PATH", "data/dashboard")

    # If path is relative, make it absolute based on project root
    if not os.path.isabs(dataset_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dataset_path = os.path.join(project_root, dataset_path)

    # Ensure directory exists
    os.makedirs(dataset_path, exist_ok=True)

    return dataset_path


@dashboard_bp.route("/dataset", methods=["GET"])
def get_dataset():
    """
    Get information about the current dashboard dataset.

    Returns:
        JSON response with dataset information or empty status
    """
    try:
        dataset_path = get_dataset_path()
        dataset_file = os.path.join(dataset_path, "dashboard_data.jsonl")

        if not os.path.exists(dataset_file):
            return jsonify({"status": "empty", "message": "No dashboard dataset found", "dataset_path": dataset_path})

        # Count records
        record_count = 0
        with open(dataset_file, "r") as f:
            for line in f:
                if line.strip():
                    record_count += 1

        # Get file size
        file_size = os.path.getsize(dataset_file)
        file_size_mb = file_size / (1024 * 1024)

        # Get file modification time
        mod_time = os.path.getmtime(dataset_file)
        from datetime import datetime

        mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")

        return jsonify(
            {
                "status": "available",
                "record_count": record_count,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size_mb, 2),
                "file_path": dataset_file,
                "last_modified": mod_time_str,
            }
        )

    except Exception as e:
        logger.error(f"Error getting dataset info: {str(e)}")
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/dataset/replace", methods=["POST"])
def replace_dataset():
    """
    Replace the dashboard dataset with new data.

    Returns:
        JSON response with operation result
    """
    try:
        dataset_path = get_dataset_path()
        dataset_file = os.path.join(dataset_path, "dashboard_data.jsonl")

        # Check if request is JSONL format or JSON list
        content_type = request.headers.get("Content-Type", "")

        if "application/jsonl" in content_type:
            # Process as JSONL
            data = jsonl_processor.load_data(request.data.decode("utf-8"))
            logger.info(f"Processing JSONL data with {len(data)} records")
        else:
            # Process as JSON list
            data = request.json
            validate_batch_request(data)
            logger.info(f"Processing JSON list with {len(data)} records")

        # Create backup of old file if it exists
        if os.path.exists(dataset_file):
            backup_file = f"{dataset_file}.bak"
            try:
                os.replace(dataset_file, backup_file)
                logger.info(f"Created backup of dashboard dataset at {backup_file}")
            except Exception as e:
                logger.warning(f"Could not create backup: {str(e)}")

        # Save the new data
        jsonl_processor.save_results(data, dataset_file)
        logger.info(f"Successfully saved {len(data)} records to {dataset_file}")

        return jsonify({"message": f"Successfully replaced dashboard dataset with {len(data)} records", "record_count": len(data), "file_path": dataset_file})

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error replacing dataset: {str(e)}")
        return jsonify({"error": str(e)}), 500


@dashboard_bp.route("/dataset/append", methods=["POST"])
def append_dataset():
    """
    Append data to the dashboard dataset.

    Returns:
        JSON response with operation result
    """
    try:
        dataset_path = get_dataset_path()
        dataset_file = os.path.join(dataset_path, "dashboard_data.jsonl")

        # Check if request is JSONL format or JSON list
        content_type = request.headers.get("Content-Type", "")

        if "application/jsonl" in content_type:
            # Process as JSONL
            data = jsonl_processor.load_data(request.data.decode("utf-8"))
            logger.info(f"Processing JSONL data with {len(data)} records")
        else:
            # Process as JSON list
            data = request.json
            validate_batch_request(data)
            logger.info(f"Processing JSON list with {len(data)} records")

        # Check if file exists
        if not os.path.exists(dataset_file):
            # Create new file with the data
            jsonl_processor.save_results(data, dataset_file)
            logger.info(f"Created new dataset file with {len(data)} records at {dataset_file}")
        else:
            # Append to existing file
            with open(dataset_file, "a") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
            logger.info(f"Appended {len(data)} records to existing dataset at {dataset_file}")

        return jsonify({"message": f"Successfully appended {len(data)} records to dashboard dataset", "records_appended": len(data), "file_path": dataset_file})

    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Error appending to dataset: {str(e)}")
        return jsonify({"error": str(e)}), 500
