import os

# Constants
APP_CONFIG_FILE = "app_config.json"
STATIC_DIR = "static"
CONFIG_DIR = "saved_configs"
EXPORT_DIR = "exports"

# Global progress state
progress_state = {
    "current": 0,
    "total": 0,
    "message": "Initializing...",
    "config": None,
    "results": None,
    "completed": False,
    "benchmark_file": None
}

# Ensure directories exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)