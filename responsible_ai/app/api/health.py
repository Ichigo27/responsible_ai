"""
Health check endpoint for the API.
"""

from flask import Blueprint, jsonify, current_app
from typing import Dict, Tuple
from logs import get_logger

logger = get_logger(__name__)
health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check() -> Tuple[Dict[str, str], int]:
    """
    Health check endpoint.
    Returns status and version information.
    """
    logger.info("Health check endpoint called")

    return jsonify({"status": "healthy", "version": current_app.config.get("VERSION", "1.0.0"), "service": "responsible_ai", "python_version": "3.13"}), 200


def register_health_endpoints(blueprint: Blueprint) -> None:
    """
    Register health check endpoints with the provided blueprint.

    Args:
        blueprint: Flask Blueprint to register endpoints with
    """
    blueprint.register_blueprint(health_bp)
