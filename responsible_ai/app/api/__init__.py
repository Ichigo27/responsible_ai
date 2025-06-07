"""
API package for the Responsible AI module.
"""

from app.api.health import health_bp
from app.api.dashboard import dashboard_bp  # Add import for dashboard blueprint
from flask import Blueprint


def create_api_blueprint(url_prefix="/api/v1"):
    """
    Create and configure the API blueprint.

    Args:
        url_prefix: URL prefix for the API endpoints

    Returns:
        Configured Flask Blueprint for the API
    """
    api_bp = Blueprint("api", __name__, url_prefix=url_prefix)

    # Register health check endpoints
    api_bp.register_blueprint(health_bp)
    
    # Register dashboard endpoints
    api_bp.register_blueprint(dashboard_bp)  # Add this line to register dashboard routes

    # Import and register endpoints (must be imported after creating the blueprint)
    from app.api.endpoints import register_api_endpoints

    register_api_endpoints(api_bp)

    return api_bp
