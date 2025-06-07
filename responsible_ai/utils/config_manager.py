"""
Configuration manager for responsible ai module.
Provides centralized configuration access.
"""

import os
import yaml
from typing import Dict, Any
from logs import get_logger


class ConfigManager:
    """
    Centralized configuration management for the Responsible AI module.
    Loads and provides access to various configuration files.
    """

    _instance = None
    _config_cache = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            # Set _base_path first before any other initialization
            cls._instance._base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
            cls._instance._config_cache = {}
            # Now it's safe to get the logger
            cls._instance.logger = get_logger(cls._instance.__class__.__name__)
            cls._instance.logger.info(f"ConfigManager initialized with base path: {cls._instance._base_path}")
        return cls._instance

    def load_config(self, config_name: str, reload: bool = False) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_name: Name of the config file without .yaml extension
            reload: Force reload the config even if cached

        Returns:
            Dict containing the configuration
        """
        if config_name in self._config_cache and not reload:
            return self._config_cache[config_name]

        config_path = os.path.join(self._base_path, f"{config_name}.yaml")
        # If file doesn't exist in base_path, try templates folder
        if not os.path.exists(config_path):
            config_path = os.path.join(self._base_path, "templates", f"{config_name}.yaml")

        if not os.path.exists(config_path):
            error_msg = f"Configuration file not found: {config_name}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            self._config_cache[config_name] = config
            self.logger.info(f"Loaded configuration: {config_name}")
            return config

    def get_config(self, config_name: str, reload: bool = False) -> Dict[str, Any]:
        """
        Get configuration by name.

        Args:
            config_name: Name of the config file without .yaml extension
            reload: Force reload the config even if cached

        Returns:
            Dict containing the configuration
        """
        return self.load_config(config_name, reload)

    def get_value(self, config_name: str, key: str, default: Any = None) -> Any:
        """
        Get a specific configuration value by key.

        Args:
            config_name: Name of the config file without .yaml extension
            key: Configuration key (supports dot notation for nested configs)
            default: Default value if key not found

        Returns:
            The configuration value or default if not found
        """
        config = self.get_config(config_name)
        keys = key.split(".")

        result = config
        try:
            for k in keys:
                result = result[k]
            return result
        except (KeyError, TypeError):
            self.logger.warning(f"Config key not found: {key} in {config_name}, using default: {default}")
            return default
