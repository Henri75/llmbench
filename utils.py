import re
import math
import json
import os
import requests
from datetime import datetime

# Server configuration file
APP_CONFIG_FILE = "app_config.json"

# Predefined size mappings for models without size in name
MODEL_SIZE_MAP = {
    'llama3.2': '3B',
    'mistral:instruct': '7B',
    'gemma2:latest': '9B',
    'qwen2.5': '7B'
}

def load_app_config():
    if os.path.exists(APP_CONFIG_FILE):
        with open(APP_CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {
        "servers": {
            "Server1": {"base_url": "http://localhost:11435/v1", "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
                        "label": "MLX"},
            "Server2": {"base_url": "http://localhost:11434/v1", "model": "llama3.2", "label": "Ollama"},
            "Server3": {"base_url": "http://localhost:11436/v1", "model": "some-model", "label": "MBP"}
        }
    }

def save_app_config(config):
    with open(APP_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def extract_short_name(model_name):
    name = model_name.lower()
    for prefix in ['mlx-community/', ':latest', ':instruct', '-instruct', '-fp16', '-q8_0', ':coder', '-grok-tool-use',
                   'alejandroolmedo/']:
        name = name.replace(prefix, '')
    name = re.sub(r'[-:.]', ' ', name).strip()
    model_families = ['llama', 'qwen', 'deepseek', 'phi', 'gemma', 'mistral', 'nomic', 'openthinker', 'tulu3']
    for part in name.split():
        for family in model_families:
            if family in part:
                return family
    return name.split()[0] if name else 'unknown'

def extract_model_version(model_name):
    matches = re.findall(r'(\d+\.\d+|\d+|r\d+|v\d+)', model_name.lower())
    return next((m for m in matches if not m.endswith('b') and not any(c in m for c in 'abcdefghijklmnopqrstuvwxyz' if c not in 'rv')), '-')

def extract_model_size(model_name, params=None):
    name = re.sub(r'(mlx-community/|alejandroolmedo/|:|-)', '.', model_name.lower())
    parts = name.split('.')
    for part in parts:
        if part.endswith('b') and part[:-1].isdigit():
            return part.upper()
    if 'mini' in name:
        return 'mini'
    for key, size in MODEL_SIZE_MAP.items():
        if key in model_name.lower():
            return size
    if params and 'B' in params:
        try:
            return f"{math.floor(float(params.replace('B', '').strip()))}B"
        except ValueError:
            pass
    return '?B'

def extract_quantization(model_name):
    matches = re.findall(r'(q\d+|fp16|fp32|\d+bit)', model_name.lower())
    if matches:
        quant = matches[-1].upper()
        return f"Q{quant.replace('BIT', '')}" if 'BIT' in quant else quant
    return 'Q4'

def get_display_name(server_name, config):
    return config.get("label", server_name)

def check_server_availability(server_name, config):
    base_url = config["base_url"]
    try:
        response = requests.get(f"{base_url}/models", timeout=5)
        response.raise_for_status()
        if response.json().get("object") == "list":
            config["api_call"] = "openai"
            print(f"{get_display_name(server_name, config)} ({base_url}): up")
            return True
    except requests.RequestException:
        try:
            response = requests.get(f"{base_url.replace('/v1', '')}/api/tags", timeout=5)
            response.raise_for_status()
            if "models" in response.json():
                config["api_call"] = "ollama"
                print(f"{get_display_name(server_name, config)} ({base_url}): up")
                return True
        except requests.RequestException:
            pass
    config["api_call"] = None
    print(f"{get_display_name(server_name, config)} ({base_url}): down")
    return False

def fetch_models(server_name, config):
    base_url = config["base_url"]
    api_call = config["api_call"]
    try:
        if api_call == "openai":
            data = requests.get(f"{base_url}/models").json()
            return data["data"] if data.get("object") == "list" else []
        elif api_call == "ollama":
            data = requests.get(f"{base_url.replace('/v1', '')}/api/tags").json()
            return [{"id": m["name"], "modified_at": m.get("modified_at", "-")} for m in data["models"]]
    except requests.RequestException as e:
        print(f"Failed to fetch models from {get_display_name(server_name, config)}: {e}")
    return []