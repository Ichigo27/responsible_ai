"""
Flask application package for Responsible AI module.
"""

from flask import Flask
from utils.config_manager import ConfigManager
from typing import Any


def create_app() -> Flask:
    """
    Application factory function.
    Creates and configures the Flask application.

    Returns:
        Flask application instance
    """
    app = Flask(__name__)

    # Load configuration
    config_manager = ConfigManager()
    app_config = config_manager.get_config("app_config")

    # Set up Flask configuration
    app.config.update(
        VERSION=app_config.get("VERSION", "1.0.0"),
        DEBUG=app_config.get("DEBUG", False),
        API_VERSION=app_config.get("API_VERSION", "v1"),
        # Add dashboard configuration to app config
        DASHBOARD=app_config.get("DASHBOARD", {}),
    )

    # Register routes
    from app.routes import register_routes

    register_routes(app)

    return app