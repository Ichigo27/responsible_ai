"""
Gunicorn configuration for the Responsible AI service.
"""

import json
from utils.config_manager import ConfigManager

# Load configuration
config_manager = ConfigManager()
app_config = config_manager.get_config("app_config")

# Gunicorn settings
bind = f"{app_config.get('HOST', '0.0.0.0')}:{app_config.get('PORT', 5000)}"

# # Set number of workers based on CPU cores
# workers_per_core = 2
# max_workers = 8
# workers = min(multiprocessing.cpu_count() * workers_per_core + 1, max_workers)

workers = 1  # For testing purposes, set to 1 worker

# Worker settings
worker_class = "gthread"
worker_connections = 1000
timeout = 120
keepalive = 5
threads = 2

# Logging
accesslog = "-"  # stdout
errorlog = "-"  # stderr
loglevel = app_config.get("LOG_LEVEL", "info").lower()

# Process naming
proc_name = "responsible_ai"

# Server mechanics
graceful_timeout = 30
max_requests = 1000
max_requests_jitter = 50

# Print configuration for logging purposes
print(
    json.dumps(
        {
            "bind": bind,
            "workers": workers,
            "worker_class": worker_class,
            "worker_connections": worker_connections,
            "timeout": timeout,
            "keepalive": keepalive,
            "loglevel": loglevel,
            "proc_name": proc_name,
        },
        indent=2,
    )
)
