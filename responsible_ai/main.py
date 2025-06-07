"""
Main entry point for the Responsible AI Evaluation Flask application.
"""

from app import create_app
from utils.config_manager import ConfigManager
from logs import get_logger

logger = get_logger(__name__)
config_manager = ConfigManager()
app_config = config_manager.get_config("app_config")

# Create Flask application
app = create_app()

if __name__ == "__main__":
    # Get configuration
    host = app_config.get("HOST")
    port = int(app_config.get("PORT"))
    debug = app_config.get("DEBUG")

    logger.info(f"Starting Responsible AI Evaluation server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
