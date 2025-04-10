from tabulate import tabulate
from datetime import datetime
import re
import math
import requests
from openai import OpenAI
import time
import json
import argparse
import statistics
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
import asyncio
import os
import logging
import pandas as pd
from io import BytesIO
import signal
import sys
from fastapi.staticfiles import StaticFiles

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server configuration file
APP_CONFIG_FILE = "app_config.json"

# Define static directory for serving static files
STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

# Load server configurations from app_config.json
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

# Save server configurations to app_config.json
def save_app_config(config):
    with open(APP_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

DEFAULT_SERVERS = load_app_config()["servers"]

# Predefined size mappings for models without size in name
MODEL_SIZE_MAP = {
    'llama3.2': '3B',
    'mistral:instruct': '7B',
    'gemma2:latest': '9B',
    'qwen2.5': '7B'
}

# Helper functions (unchanged)
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
    return next((m for m in matches if
                 not m.endswith('b') and not any(c in m for c in 'abcdefghijklmnopqrstuvwxyz' if c not in 'rv')), '-')

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

# Server and model utilities (unchanged)
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

def benchmark(config, messages, server_name, repeat, total_repeats, progress_state):
    progress_state[
        "message"] = f"Benchmarking {get_display_name(server_name, config)} - Repeat {repeat}/{total_repeats}"
    logger.info(
        f"Progress: current={progress_state['current']}, total={progress_state['total']}, message={progress_state['message']}")
    print(
        f"\n{get_display_name(server_name, config)} Server - Repeat {repeat}/{total_repeats} - Model: {config['model']}")
    start_time = time.time()
    ttft, output, total_tokens = None, "", 0
    try:
        print("* ", end="", flush=True)
        if config["api_call"] == "openai":
            client = OpenAI(base_url=config["base_url"], api_key="pyomlx")
            response = client.chat.completions.create(model=config["model"], messages=messages, stream=True)
            for chunk in response:
                if not ttft and hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content:
                    ttft = time.time() - start_time
                if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content:
                    output += chunk.choices[0].delta.content
            total_tokens = len(output.split())
        elif config["api_call"] == "ollama":
            response = requests.post(f"{config['base_url'].replace('/v1', '')}/api/chat",
                                     json={"model": config["model"], "messages": messages}, stream=True)
            first_chunk = True
            for line in response.iter_lines():
                if line and (data := json.loads(line)) and "message" in data:
                    if first_chunk:
                        ttft = time.time() - start_time
                        first_chunk = False
                    output += data["message"]["content"]
                    total_tokens += len(data["message"]["content"].split())
    except Exception as e:
        print(f"Failed to benchmark {get_display_name(server_name, config)}: {e}")
        return "", 0, 0, 0, 0, 0
    total_time = time.time() - start_time
    ttft = ttft or total_time
    gen_time = max(total_time - ttft, 0)
    tokens_per_sec = total_tokens / gen_time if gen_time > 0 else 0
    return output, total_time, ttft, gen_time, total_tokens, tokens_per_sec

# Shared utilities (unchanged)
def get_available_servers(server_configs):
    if server_configs is None:
        return {}
    return {name: config for name, config in server_configs.items() if check_server_availability(name, config)}

def select_servers(available_servers, num_servers, interactive=False, combo=None):
    if not available_servers:
        raise ValueError("No available servers provided")

    from itertools import combinations
    server_list = list(available_servers.keys())
    combos = []
    if num_servers <= len(server_list):
        combos = ['+'.join(combo) for combo in combinations(server_list, num_servers)]
    else:
        raise ValueError(f"Requested {num_servers} servers, but only {len(server_list)} available")

    display_map = {s: get_display_name(s, available_servers[s]) for s in available_servers}

    if interactive:
        print(f"\nAvailable combinations for {num_servers} server(s):")
        display_combos = ['+'.join(display_map[s] for s in c.split('+')) for c in combos]
        for i, c in enumerate(display_combos, 1):
            print(f"{i}. {c}")
        choice = int(input(f"Choose a combination (1-{len(combos)}) [default: 1]: ") or 1) - 1
        return [(s, available_servers[s]) for s in combos[choice].split('+')], display_combos[choice]
    if combo in combos:
        return [(s, available_servers[s]) for s in combo.split('+')], '+'.join(display_map[s] for s in combo.split('+'))
    raise ValueError(f"Invalid server combination: {combo}")

def display_model_table(all_models):
    table_data = []
    for server_name, models in all_models.items():
        for model in models:
            row = [model['server'], model['model_name'], model['model_version'], model['model_size'],
                   model['quantization'], model['instruct']]
            for s in ['Server1', 'Server2', 'Server3']:
                row.extend(
                    [model['name'] if server_name == s else '-', model['timestamp'] if server_name == s else '-'])
            table_data.append(row)
    headers = ['Server', 'Model Name', 'Model Version', 'Model Size', 'Quantization', 'Instruct',
               'Server1 Model', 'Server1 Created', 'Server2 Model', 'Server2 Modified', 'Server3 Model',
               'Server3 Created']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))

def find_common_models(selected_servers, all_models):
    server_names = [s[0] for s in selected_servers]
    common_models, seen = [], set()
    for m1 in all_models[server_names[0]]:
        key = (m1['model_name'], m1['model_version'], m1['model_size'], m1['quantization'])
        if key in seen:
            continue
        is_common = True
        entry = {'model_name': m1['model_name'], 'model_version': m1['model_version'], 'model_size': m1['model_size'],
                 'quantization': m1['quantization']}
        for sn in server_names[1:]:
            found = False
            for m in all_models[sn]:
                if (m['model_name'] == m1['model_name'] and m['model_version'] == m1['model_version'] and
                        m['model_size'] == m1['model_size'] and m['quantization'] == m1['quantization']):
                    found = True
                    entry[f'{sn}_name'] = m['name']
                    entry[f'instruct_{sn}'] = m['instruct']
                    break
            if not found:
                is_common = False
                break
        if is_common:
            entry[f'{server_names[0]}_name'] = m1['name']
            entry[f'instruct_{server_names[0]}'] = m1['instruct']
            instruct_vals = [entry[f'instruct_{s}'] for s in server_names]
            entry['match_type'] = 'perfect' if all(v == instruct_vals[0] for v in instruct_vals) else 'partial'
            common_models.append(entry)
            seen.add(key)
    return sorted(common_models, key=lambda x: (x['model_name'], x['model_version']))

def run_benchmarks(selected_servers, messages, num_repeats, progress_state, base_steps=0):
    total_steps_per_benchmark = len(selected_servers) * num_repeats
    results = {}
    step_increment = 0
    for name, config in selected_servers:
        outputs, total_times, ttfts, gen_times, tokens, tokens_per_sec = [], [], [], [], [], []
        print(f"\nStarting benchmark for {get_display_name(name, config)} server...")
        for i in range(num_repeats):
            step_increment += 1
            progress_state["current"] = base_steps + step_increment
            out, tt, ttf, gt, tok, tps = benchmark(config, messages, name, i + 1, num_repeats, progress_state)
            outputs.append(out)
            total_times.append(tt)
            ttfts.append(ttf)
            gen_times.append(gt)
            tokens.append(tok)
            tokens_per_sec.append(tps)
        results[name] = {
            'display_name': get_display_name(name, config),
            'full_model_name': config["model"],
            'outputs': outputs,
            'total_times': total_times,
            'ttfts': ttfts,
            'generation_times': gen_times,
            'total_tokens_list': tokens,
            'tokens_per_sec_list': tokens_per_sec
        }
    return results, total_steps_per_benchmark

def generate_stats_table(results, interactive=True, bench_id=None):
    table, headers = [], []
    for name, data in results.items():
        tt, ttf, gt, tps = data['total_times'], data['ttfts'], data['generation_times'], data['tokens_per_sec_list']
        stats = lambda x: (
        min(x) if x else 0, max(x) if x else 0, sum(x) / len(x) if x else 0, statistics.stdev(x) if len(x) > 1 else 0)
        tt_stats, ttf_stats, gt_stats, tps_stats = map(stats, [tt, ttf, gt, tps])
        last_output = data['outputs'][-1] if data['outputs'] else "No output"
        row = [data['display_name'], data['full_model_name']]
        if not interactive:
            row = ([f"Benchmark {bench_id}"] if bench_id else []) + row + [
                extract_short_name(data['full_model_name']),
                extract_model_version(data['full_model_name']),
                extract_model_size(data['full_model_name']),
                extract_quantization(data['full_model_name'])
            ]
        row.extend([f"{v:.2f}" for stat in (tt_stats, ttf_stats, gt_stats, tps_stats) for v in
                    (stat[:3] if interactive else stat)])
        table.append((row, last_output))
    headers = ["Server", "Full Model Name"] if interactive else ["Benchmark", "Server", "Full Model Name", "Model Name",
                                                                 "Version", "Size", "Quantization"]
    headers.extend(
        sum([["Min Total (s)", "Max Total (s)", "Avg Total (s)"] + (["Std Total (s)"] if not interactive else []),
             ["Min TTFT (s)", "Max TTFT (s)", "Avg TTFT (s)"] + (["Std TTFT (s)"] if not interactive else []),
             ["Min Gen (s)", "Max Gen (s)", "Avg Gen (s)"] + (["Std Gen (s)"] if not interactive else []),
             ["Min Tokens/s", "Max Tokens/s", "Avg Tokens/s"] + (["Std Tokens/s"] if not interactive else [])], []))
    return table, headers

# Main modes (unchanged)
def run_interactive_mode():
    servers = get_available_servers(DEFAULT_SERVERS.copy())
    if not servers:
        print("No servers available.")
        exit(1)
    print(f"\n{len(servers)} servers available for benchmark.")
    num_servers = min(int(input(f"\nHow many servers to benchmark (1-{len(servers)})? [default: 1]: ") or 1), len(servers))
    selected_servers, combo_display = select_servers(servers, num_servers, True)

    all_models = {name: [{
        'server': get_display_name(name, config),
        'model_name': extract_short_name(m['id']),
        'model_version': extract_model_version(m['id']),
        'model_size': extract_model_size(m['id']),
        'quantization': extract_quantization(m['id']),
        'instruct': 'yes' if 'instruct' in m['id'].lower() else 'no',
        'name': m['id'],
        'timestamp': datetime.fromtimestamp(m.get('created', time.time())).isoformat() + '+00:00' if config["api_call"] == "openai" and isinstance(m.get('created'), (int, float)) else m.get('modified_at', '-')
    } for m in fetch_models(name, config)] for name, config in selected_servers}
    for name, config in selected_servers:
        print(f"{get_display_name(name, config)}: {len(all_models[name])} models")
    display_model_table(all_models)

    common_models = find_common_models(selected_servers, all_models)
    print("\nCommon Models (based on name, version, size, and quantization, with optional instruct match):")
    if common_models:
        for i, m in enumerate(common_models, 1):
            flag = " (!)" if m['match_type'] == 'partial' else ""
            servers_display = '+'.join(get_display_name(s[0], s[1]) for s in selected_servers)
            print(f"{i}. {m['model_name']} {m['model_version']} {m['model_size']} {m['quantization']}{flag} ({servers_display})")
    else:
        print("No common models found. Selecting models individually.")

    selected_models = {}
    if common_models:
        choice = input(f"\nSelect a common model for benchmarking by entering its number (1-{len(common_models)}), or 0 to select individually [default: 1]: ") or "1"
        if choice.isdigit() and 1 <= int(choice) <= len(common_models):
            model = common_models[int(choice) - 1]
            selected_models = {name: model[f'{name}_name'] for name, _ in selected_servers}
            for name, config in selected_servers:
                config["model"] = selected_models[name]
        else:
            choice = "0"
    if not common_models or choice == "0":
        print("\nSelecting models individually for each server:")
        for name, config in selected_servers:
            models = all_models[name]
            for i, m in enumerate(models, 1):
                print(f"{i}. {m['name']} ({m['model_name']} {m['model_version']} {m['model_size']} {m['quantization']})")
            model_choice = int(input(f"Select a model for {get_display_name(name, config)} (1-{len(models)}) [default: 1]: ") or 1) - 1
            selected_models[name] = models[model_choice]['name']
            config["model"] = selected_models[name]

    messages = [{'role': 'user', 'content': input("Enter your custom prompt: ") if input("\nDo you want to use a custom prompt? (yes/no) [default: no]: ").lower() in ('yes', 'y') else "write a 200 words story"}]
    num_repeats = max(int(input("Enter the number of benchmark repeats [default: 1]: ") or 1), 1)
    progress_state = {"current": 0, "total": len(selected_servers) * num_repeats, "message": ""}
    results, _ = run_benchmarks(selected_servers, messages, num_repeats, progress_state)

    summary = [{"Benchmark": "Interactive", "Servers": combo_display,
                "Model": ", ".join(f"{get_display_name(n, c)}: {c['model']}" for n, c in selected_servers),
                "Query": messages[0]['content'], "Repeats": num_repeats}]
    print("\n--- Benchmark Queries Summary ---\n" + tabulate(
        [[d[h] for h in "Benchmark Servers Model Query Repeats".split()] for d in summary],
        headers="Benchmark Servers Model Query Repeats".split(), tablefmt='grid'))
    print("\n--- Benchmark Results ---")
    table, headers = generate_stats_table(results)
    print(tabulate([row for row, _ in table], headers=headers, tablefmt='grid'))
    return [(1, {}, selected_servers, results)], summary

def run_non_interactive_mode(config_file):
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' does not exist.")
        exit(1)
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{config_file}': {str(e)}")
        exit(1)

    servers = DEFAULT_SERVERS.copy()
    if config.get("servers") is not None:
        servers.update(config["servers"])

    progress_state = {"current": 0, "total": 0, "message": ""}
    total_steps = sum(len(get_available_servers(servers.copy().update(b.get("servers", {}) or {}))) * b.get("num_repeats", 1) for b in config["benchmarks"])
    progress_state["total"] = total_steps
    results_list, summaries = [], []
    base_steps = 0
    for idx, bench in enumerate(config["benchmarks"], 1):
        bench_servers = servers.copy()
        if bench.get("servers") is not None:
            bench_servers.update(bench["servers"])
        available = get_available_servers(bench_servers)
        if not available or not (1 <= (num_servers := bench.get("num_servers", 0)) <= len(available)):
            print(f"Benchmark {idx}: Invalid setup. Skipping.")
            continue

        try:
            selected_servers, combo_display = select_servers(available, num_servers, False, bench.get("server_combo"))
        except ValueError as e:
            print(f"Benchmark {idx}: {str(e)}. Skipping.")
            continue

        all_models = {name: [{
            'server': get_display_name(name, config),
            'model_name': extract_short_name(m['id']),
            'model_version': extract_model_version(m['id']),
            'model_size': extract_model_size(m['id']),
            'quantization': extract_quantization(m['id']),
            'instruct': 'yes' if 'instruct' in m['id'].lower() else 'no',
            'name': m['id'],
            'timestamp': datetime.fromtimestamp(m.get('created', time.time())).isoformat() + '+00:00' if config["api_call"] == "openai" and isinstance(m.get('created'), (int, float)) else m.get('modified_at', '-')
        } for m in fetch_models(name, config)] for name, config in selected_servers}

        selected_models = {}
        if "models" in bench:
            if not bench["models"]:
                print(f"Benchmark {idx}: No models specified. Skipping.")
                continue
            for s, m in bench["models"].items():
                if s in [n for n, _ in selected_servers] and any(mod['name'] == m for mod in all_models.get(s, [])):
                    selected_models[s] = m
                else:
                    print(f"Benchmark {idx}: Model {m} not found for {s}. Skipping.")
                    break
            else:
                for n, c in selected_servers:
                    if n in selected_models:
                        c["model"] = selected_models[n]
        elif "model_name" in bench:
            target = bench["model_name"].lower().split()
            if len(target) == 4:
                for n, c in selected_servers:
                    match = next((m for m in all_models[n] if [m[k].lower() for k in ['model_name', 'model_version', 'model_size', 'quantization']] == target), None)
                    if match:
                        selected_models[n] = match['name']
                        c["model"] = match['name']
                    else:
                        print(f"Benchmark {idx}: No match for {bench['model_name']} on {get_display_name(n, c)}. Skipping.")
                        break
            else:
                print(f"Benchmark {idx}: Invalid model_name format. Skipping.")
                continue
        if len(selected_models) != len(selected_servers):
            print(f"Benchmark {idx}: Failed to select models for all servers. Skipping.")
            continue

        messages = [{'role': 'user', 'content': bench.get('custom_prompt', 'write a 200 words story')}]
        num_repeats = max(bench.get('num_repeats', 1), 1)
        results, steps = run_benchmarks(selected_servers, messages, num_repeats, progress_state, base_steps)
        base_steps += steps
        results_list.append((idx, bench, selected_servers, results))
        summaries.append({"Benchmark": f"Benchmark {idx}", "Servers": combo_display,
                         "Model": ", ".join(f"{get_display_name(n, c)}: {c['model']}" for n, c in selected_servers),
                         "Query": messages[0]['content'], "Repeats": num_repeats})

    print("\n--- Benchmark Queries Summary ---\n" + tabulate(
        [[d[h] for h in "Benchmark Servers Model Query Repeats".split()] for d in summaries],
        headers="Benchmark Servers Model Query Repeats".split(), tablefmt='grid'))
    print("\n--- Enhanced Benchmark Results ---")
    all_table = []
    for idx, _, _, res in results_list:
        table, headers = generate_stats_table(res, False, idx)
        all_table.extend([row for row, _ in table])
    print(tabulate(all_table, headers=headers, tablefmt='grid'))
    return results_list, summaries

# Server mode components
app = FastAPI(title="LLM Benchmark Server")

# Mount static directory to serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class BenchmarkConfig(BaseModel):
    config: dict

class ServerConfig(BaseModel):
    name: str
    base_url: str
    label: str
    model: str = None

progress_state = {
    "current": 0,
    "total": 0,
    "message": "Initializing...",
    "config": None,
    "results": None,
    "completed": False,
    "benchmark_file": None
}

CONFIG_DIR = "saved_configs"
os.makedirs(CONFIG_DIR, exist_ok=True)
EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)


@app.get("/progress")
async def get_progress():
    return {
        "current": progress_state["current"],
        "total": progress_state["total"],
        "message": progress_state["message"],
        "completed": progress_state["completed"],
        "results": progress_state["results"] if progress_state["completed"] else None
    }

@app.get("/servers")
async def get_servers():
    try:
        config = load_app_config()
        servers = config["servers"]
        global DEFAULT_SERVERS
        DEFAULT_SERVERS = servers
        available_servers = get_available_servers(servers.copy())
        if not available_servers:
            return {"error": "No available servers found"}

        result = {}
        for name, server_config in servers.items():
            status = "up" if name in available_servers else "down"
            result[name] = {
                "label": server_config["label"],
                "base_url": server_config["base_url"],
                "api_call": available_servers.get(name, {}).get("api_call", None),
                "status": status
            }
        return result
    except Exception as e:
        logger.error(f"Error fetching servers: {str(e)}")
        return {"error": "Failed to load server configuration"}

@app.get("/app_config")
async def get_app_config():
    try:
        config = load_app_config()
        servers = config["servers"]
        available_servers = get_available_servers(servers.copy())
        for name, server_config in servers.items():
            server_config["status"] = "up" if name in available_servers else "down"
            server_config["api_call"] = available_servers.get(name, {}).get("api_call", None)
        return config
    except Exception as e:
        logger.error(f"Error fetching app config: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load app configuration")

@app.post("/app_config")
async def update_app_config(config: dict):
    save_app_config(config)
    global DEFAULT_SERVERS
    DEFAULT_SERVERS = config["servers"]
    return {"message": "App config updated successfully"}

@app.get("/models")
async def get_models(server: str):
    servers = get_available_servers(DEFAULT_SERVERS.copy())
    if server not in servers:
        raise HTTPException(status_code=404, detail="Server not found")
    models = fetch_models(server, servers[server])
    parsed_models = [{
        'full_name': m["id"],
        'internal_name': f"{extract_short_name(m['id'])} {extract_model_version(m['id'])} {extract_model_size(m['id'])} {extract_quantization(m['id'])}",
        'modified_at': m.get("modified_at", "-")
    } for m in models]
    return sorted(parsed_models, key=lambda x: x['internal_name'])

@app.get("/common_models")
async def get_common_models(servers: str):
    server_list = servers.split(',')
    if len(server_list) < 2:
        raise HTTPException(status_code=400, detail="At least two servers are required for common models")
    available_servers = get_available_servers(DEFAULT_SERVERS.copy())
    selected_servers = [(s, available_servers[s]) for s in server_list if s in available_servers]
    if len(selected_servers) != len(server_list):
        raise HTTPException(status_code=404, detail="One or more servers not found or unavailable")

    all_models = {name: [{
        'server': get_display_name(name, config),
        'model_name': extract_short_name(m['id']),
        'model_version': extract_model_version(m['id']),
        'model_size': extract_model_size(m['id']),
        'quantization': extract_quantization(m['id']),
        'instruct': 'yes' if 'instruct' in m['id'].lower() else 'no',
        'name': m['id'],
        'timestamp': m.get('modified_at', '-')
    } for m in fetch_models(name, config)] for name, config in selected_servers}

    common_models = find_common_models(selected_servers, all_models)
    return [{
        'internal_name': f"{m['model_name']} {m['model_version']} {m['model_size']} {m['quantization']}",
        'server_models': {s[0]: m[f'{s[0]}_name'] for s in selected_servers}
    } for m in common_models]

@app.post("/benchmark")
async def run_benchmark(config: BenchmarkConfig):
    try:
        progress_state["config"] = config.config
        progress_state["current"] = 0
        servers = DEFAULT_SERVERS.copy()
        if config.config.get("servers") is not None:
            servers.update(config.config["servers"])
        total_steps = 0
        for bench in config.config["benchmarks"]:
            bench_servers = servers.copy()
            if bench.get("servers") is not None:
                bench_servers.update(bench["servers"])
            available = get_available_servers(bench_servers)
            num_servers = bench.get("num_servers", len(available))
            num_repeats = max(bench.get("num_repeats", 1), 1)
            total_steps += num_servers * num_repeats
        progress_state["total"] = total_steps
        progress_state["message"] = "Starting benchmarks..."
        progress_state["completed"] = False
        results, summaries = await asyncio.to_thread(run_non_interactive_mode_from_config, config.config)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"benchmark_results_{timestamp}.txt"
        with open(output_file, 'w') as f:
            f.write("==================================================\n")
            f.write("          LLM Benchmark Report\n")
            f.write(f"          Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("==================================================\n\n")
            f.write("--- Benchmark Parameters Summary ---\n\n")
            params_table = []
            for idx, bench in enumerate(config.config["benchmarks"], 1):
                bench_servers = servers.copy()
                if bench.get("servers") is not None:
                    bench_servers.update(bench["servers"])
                available = get_available_servers(bench_servers)
                selected_servers, _ = select_servers(available, bench.get("num_servers", len(available)), False,
                                                     bench.get("server_combo"))
                models_str = "Common Model: " + bench.get("model_name", "N/A") if "model_name" in bench else ", ".join(
                    f"{get_display_name(n, c)}: {c['model']}" for n, c in selected_servers)
                params_table.append([f"Benchmark {idx}", len(selected_servers), models_str, bench.get("num_repeats", 1),
                                     bench.get("custom_prompt", "write a 200 words story")])
            f.write(tabulate(params_table,
                             headers=["Benchmark", "Number of Servers", "Models", "Number of Iterations", "LLM Query"],
                             tablefmt='grid'))
            f.write("\n\n")
            f.write("--- Servers and Models Details ---\n\n")
            servers_table = []
            for idx, _, selected_servers, _ in results:
                for name, config in selected_servers:
                    servers_table.append([f"Benchmark {idx}", get_display_name(name, config), config["base_url"],
                                          extract_short_name(config["model"]), extract_model_version(config["model"]),
                                          extract_model_size(config["model"]), extract_quantization(config["model"]),
                                          config["model"]])
            f.write(tabulate(servers_table,
                             headers=["Benchmark", "Server Label", "Base URL", "Model Short Name", "Model Version",
                                      "Model Size", "Quantization", "Full Model Name"], tablefmt='grid'))
            f.write("\n\n")
            f.write("--- Detailed Benchmark Results ---\n\n")
            for idx, _, selected_servers, res in results:
                f.write(f"===== Benchmark {idx} =====\n\n")
                for name, data in res.items():
                    f.write(f"Server: {data['display_name']}\n")
                    f.write("-" * 50 + "\n")
                    server_details = [
                        ["Label", data["display_name"]],
                        ["Base URL", next(c["base_url"] for n, c in selected_servers if n == name)],
                        ["Model Short Name", extract_short_name(data["full_model_name"])],
                        ["Model Version", extract_model_version(data["full_model_name"])],
                        ["Model Size", extract_model_size(data["full_model_name"])],
                        ["Quantization", extract_quantization(data["full_model_name"])],
                        ["Full Model Name", data["full_model_name"]]
                    ]
                    f.write(tabulate(server_details, headers=["Parameter", "Value"], tablefmt='grid'))
                    f.write("\n\n")
                    f.write("Pass Details:\n")
                    pass_table = []
                    for i, (output, tt, ttf, gt, tok, tps) in enumerate(
                            zip(data['outputs'], data['total_times'], data['ttfts'], data['generation_times'],
                                data['total_tokens_list'], data['tokens_per_sec_list']), 1):
                        pass_table.append([i, f"{tt:.2f}", f"{ttf:.2f}", f"{gt:.2f}", tok, f"{tps:.2f}"])
                        f.write(f"Pass {i} Output:\n")
                        f.write("```\n" + (output or "No output generated") + "\n```\n\n")
                    f.write(tabulate(pass_table,
                                     headers=["Pass", "Total Time (s)", "TTFT (s)", "Generation Time (s)",
                                              "Total Tokens", "Tokens/s"], tablefmt='grid'))
                    f.write("\n\n")
            f.write("--- Global Benchmark Results Summary ---\n\n")
            global_table = []
            for r in results:
                idx = r[0]
                table, headers = generate_stats_table(r[3], interactive=False, bench_id=idx)
                for row, _ in table:
                    global_table.append(row)
            f.write(tabulate(global_table, headers=headers, tablefmt='grid'))
            f.write("\n==================================================\n")

        progress_state["benchmark_file"] = output_file
        progress_state["results"] = {
            "summaries": summaries,
            "results": [{"benchmark_id": r[0],
                         "servers": {name: {"display_name": r[3][name]["display_name"],
                                            "model": r[3][name]["full_model_name"]} for name in r[3]},
                         "stats": generate_stats_table(r[3], interactive=False, bench_id=r[0])[0],
                         "full_results": r[3]} for r in results],
            "benchmark_file": output_file
        }
        progress_state["current"] = progress_state["total"]
        progress_state["message"] = "Completed"
        progress_state["completed"] = True
        return progress_state["results"]
    except Exception as e:
        progress_state["message"] = f"Error: {str(e)}"
        progress_state["completed"] = True
        raise HTTPException(status_code=400, detail=f"Error running benchmark: {str(e)}")

@app.get("/benchmark_file")
async def get_benchmark_file():
    if not progress_state["benchmark_file"] or not os.path.exists(progress_state["benchmark_file"]):
        raise HTTPException(status_code=404, detail="Benchmark file not found")
    with open(progress_state["benchmark_file"], 'r') as f:
        return {"content": f.read()}

@app.get("/export_excel/{benchmark_id}")
async def export_excel(benchmark_id: str):
    try:
        if not progress_state["results"] or not progress_state["completed"]:
            logger.error("No benchmark results available for export")
            raise HTTPException(status_code=404, detail="No benchmark results available. Please run a benchmark first.")

        if benchmark_id == "all":
            df_data = []
            for result in progress_state["results"]["results"]:
                table, headers = generate_stats_table(result["full_results"], interactive=False, bench_id=result["benchmark_id"])
                for row, _ in table:
                    df_data.append(row)
            df = pd.DataFrame(df_data, columns=headers)
            logger.info(f"Generated data for all benchmarks: {len(df_data)} rows")
        else:
            bench = next((r for r in progress_state["results"]["results"] if str(r["benchmark_id"]) == benchmark_id), None)
            if not bench:
                logger.error(f"Benchmark ID {benchmark_id} not found")
                raise HTTPException(status_code=404, detail=f"Benchmark ID {benchmark_id} not found")
            table, headers = generate_stats_table(bench["full_results"], interactive=False, bench_id=bench["benchmark_id"])
            df_data = [row for row, _ in table]
            df = pd.DataFrame(df_data, columns=headers)
            logger.info(f"Generated data for benchmark {benchmark_id}: {len(df_data)} rows")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{benchmark_id}_{timestamp}.xlsx"
        filepath = os.path.join(EXPORT_DIR, filename)

        os.makedirs(EXPORT_DIR, exist_ok=True)
        logger.info(f"Export directory ensured: {EXPORT_DIR}")

        try:
            df.to_excel(filepath, index=False)
            logger.info(f"Excel file saved successfully: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save Excel file {filepath}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save Excel file: {str(e)}")

        if not os.path.exists(filepath):
            logger.error(f"Excel file {filepath} was not created")
            raise HTTPException(status_code=500, detail="Excel file creation failed")

        return FileResponse(filepath, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            filename=filename)

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in export_excel: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/saved_configs")
async def list_saved_configs():
    configs = [f for f in os.listdir(CONFIG_DIR) if f.endswith('.json')]
    return {"configs": configs}

@app.post("/save_config")
async def save_config(config: dict):
    name = config.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Config name is required")
    filename = os.path.join(CONFIG_DIR, f"{name}.json")
    with open(filename, 'w') as f:
        json.dump(config["config"], f, indent=2)
    return {"message": f"Config saved as {filename}"}

@app.get("/load_config/{name}")
async def load_config(name: str):
    filename = os.path.join(CONFIG_DIR, f"{name}.json")
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="Config not found")
    with open(filename, 'r') as f:
        return json.load(f)

@app.delete("/delete_config/{name}")
async def delete_config(name: str):
    filename = os.path.join(CONFIG_DIR, f"{name}.json")
    if not os.path.exists(filename):
        raise HTTPException(status_code=404, detail="Config not found")
    os.remove(filename)
    return {"message": f"Config {name} deleted"}

@app.get("/")
async def get_web_interface():
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, STATIC_DIR, "index.html")
    return FileResponse(file_path)

def run_non_interactive_mode_from_config(config_data):
    servers = DEFAULT_SERVERS.copy()
    if config_data.get("servers") is not None:
        servers.update(config_data["servers"])
    results_list, summaries = [], []
    base_steps = 0
    progress_state["current"] = 0
    total_steps = 0
    for bench in config_data["benchmarks"]:
        bench_servers = servers.copy()
        if bench.get("servers") is not None:
            bench_servers.update(bench["servers"])
        available = get_available_servers(bench_servers)
        num_servers = bench.get("num_servers", len(available))
        num_repeats = max(bench.get("num_repeats", 1), 1)
        total_steps += num_servers * num_repeats
    progress_state["total"] = total_steps

    for idx, bench in enumerate(config_data["benchmarks"], 1):
        progress_state["message"] = f"Setting up Benchmark {idx}"
        logger.info(f"Progress: current={progress_state['current']}, total={progress_state['total']}, message={progress_state['message']}")
        bench_servers = servers.copy()
        if bench.get("servers") is not None:
            bench_servers.update(bench["servers"])
        available = get_available_servers(bench_servers)
        if not available or not (1 <= (num_servers := bench.get("num_servers", 0)) <= min(3, len(available))):
            print(f"Benchmark {idx}: Invalid setup. Skipping.")
            continue
        try:
            selected_servers, combo_display = select_servers(available, num_servers, False, bench.get("server_combo"))
        except ValueError as e:
            print(f"Benchmark {idx}: {str(e)}. Skipping.")
            continue
        progress_state["message"] = f"Fetching models for Benchmark {idx}"
        logger.info(f"Progress: current={progress_state['current']}, total={progress_state['total']}, message={progress_state['message']}")
        all_models = {name: [{
            'server': get_display_name(name, config),
            'model_name': extract_short_name(m['id']),
            'model_version': extract_model_version(m['id']),
            'model_size': extract_model_size(m['id']),
            'quantization': extract_quantization(m['id']),
            'instruct': 'yes' if 'instruct' in m['id'].lower() else 'no',
            'name': m['id'],
            'timestamp': datetime.fromtimestamp(m.get('created', time.time())).isoformat() + '+00:00' if config["api_call"] == "openai" and isinstance(m.get('created'), (int, float)) else m.get('modified_at', '-')
        } for m in fetch_models(name, config)] for name, config in selected_servers}
        selected_models = {}
        if "models" in bench:
            if not bench["models"]:
                print(f"Benchmark {idx}: No models specified. Skipping.")
                continue
            for s, m in bench["models"].items():
                if s in [n for n, _ in selected_servers] and any(mod['name'] == m for mod in all_models.get(s, [])):
                    selected_models[s] = m
                else:
                    print(f"Benchmark {idx}: Model {m} not found for {s}. Skipping.")
                    break
            else:
                for n, c in selected_servers:
                    if n in selected_models:
                        c["model"] = selected_models[n]
        elif "model_name" in bench:
            target = bench["model_name"].lower().split()
            if len(target) == 4:
                for n, c in selected_servers:
                    match = next((m for m in all_models[n] if [m[k].lower() for k in ['model_name', 'model_version', 'model_size', 'quantization']] == target), None)
                    if match:
                        selected_models[n] = match['name']
                        c["model"] = match['name']
                    else:
                        print(f"Benchmark {idx}: No match for {bench['model_name']} on {get_display_name(n, c)}. Skipping.")
                        break
            else:
                print(f"Benchmark {idx}: Invalid model_name format. Skipping.")
                continue
        if len(selected_models) != len(selected_servers):
            print(f"Benchmark {idx}: Failed to select models for all servers. Skipping.")
            continue
        messages = [{"role": "user", "content": bench.get("custom_prompt", "write a 200 words story")}]
        num_repeats = max(bench.get("num_repeats", 1), 1)
        progress_state["message"] = f"Running Benchmark {idx}"
        logger.info(f"Progress: current={progress_state['current']}, total={progress_state['total']}, message={progress_state['message']}")
        results, steps = run_benchmarks(selected_servers, messages, num_repeats, progress_state, base_steps)
        base_steps += steps
        results_list.append((idx, bench, selected_servers, results))
        summaries.append({"Benchmark": f"Benchmark {idx}", "Servers": combo_display,
                          "Model": ", ".join(f"{get_display_name(n, c)}: {c['model']}" for n, c in selected_servers),
                          "Query": messages[0]["content"], "Repeats": num_repeats})
    return results_list, summaries

def run_server_mode():
    print("Starting LLM Benchmark Server on http://localhost:8086")
    def shutdown(sig, frame):
        print("\nReceived shutdown signal. Exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    uvicorn.run(app, host="127.0.0.1", port=8086)

# Main execution (unchanged)
parser = argparse.ArgumentParser(description="Run benchmarks")
parser.add_argument('-f', '--file', type=str, help="Config JSON file for non-interactive mode")
parser.add_argument('-s', '--server', action='store_true', help="Run in server mode with FastAPI")
args = parser.parse_args()

if args.server:
    run_server_mode()
else:
    results, summaries = run_non_interactive_mode(args.file) if args.file else run_interactive_mode()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_{timestamp}.txt"
    with open(output_file, 'w') as f:
        f.write("==================================================\n")
        f.write("          LLM Benchmark Report\n")
        f.write(f"          Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("==================================================\n\n")
        f.write("--- Benchmark Parameters Summary ---\n\n")
        params_table = []
        for r in results:
            idx, bench, selected_servers, _ = r
            models_str = "Common Model: " + bench.get("model_name", "N/A") if "model_name" in bench else ", ".join(
                f"{get_display_name(n, c)}: {c['model']}" for n, c in selected_servers)
            params_table.append(
                [f"Benchmark {idx}" if len(results) > 1 else "Interactive", len(selected_servers), models_str,
                 bench.get("num_repeats", 1), summaries[0]["Query"]])
        f.write(tabulate(params_table, headers=["Benchmark", "Number of Servers", "Models", "Number of Iterations", "LLM Query"], tablefmt='grid'))
        f.write("\n\n")
        f.write("--- Servers and Models Details ---\n\n")
        servers_table = []
        for idx, _, selected_servers, _ in results:
            for name, config in selected_servers:
                servers_table.append([
                    f"Benchmark {idx}" if len(results) > 1 else "Interactive",
                    get_display_name(name, config),
                    config["base_url"],
                    extract_short_name(config["model"]),
                    extract_model_version(config["model"]),
                    extract_model_size(config["model"]),
                    extract_quantization(config["model"]),
                    config["model"]
                ])
        f.write(tabulate(servers_table, headers=["Benchmark", "Server Label", "Base URL", "Model Short Name", "Model Version", "Model Size", "Quantization", "Full Model Name"], tablefmt='grid'))
        f.write("\n\n")
        f.write("--- Detailed Benchmark Results ---\n\n")
        for idx, _, selected_servers, res in results:
            f.write(f"===== Benchmark {idx if len(results) > 1 else 'Interactive'} =====\n\n")
            for name, data in res.items():
                f.write(f"Server: {data['display_name']}\n")
                f.write("-" * 50 + "\n")
                server_details = [
                    ["Label", data["display_name"]],
                    ["Base URL", next(c["base_url"] for n, c in selected_servers if n == name)],
                    ["Model Short Name", extract_short_name(data["full_model_name"])],
                    ["Model Version", extract_model_version(data["full_model_name"])],
                    ["Model Size", extract_model_size(data["full_model_name"])],
                    ["Quantization", extract_quantization(data["full_model_name"])],
                    ["Full Model Name", data["full_model_name"]]
                ]
                f.write(tabulate(server_details, headers=["Parameter", "Value"], tablefmt='grid'))
                f.write("\n\n")
                f.write("Pass Details:\n")
                pass_table = []
                for i, (output, tt, ttf, gt, tok, tps) in enumerate(
                        zip(data['outputs'], data['total_times'], data['ttfts'], data['generation_times'],
                            data['total_tokens_list'], data['tokens_per_sec_list']), 1):
                    pass_table.append([i, f"{tt:.2f}", f"{ttf:.2f}", f"{gt:.2f}", tok, f"{tps:.2f}"])
                    f.write(f"Pass {i} Output:\n")
                    f.write("```\n" + (output or "No output generated") + "\n```\n\n")
                f.write(tabulate(pass_table,
                                 headers=["Pass", "Total Time (s)", "TTFT (s)", "Generation Time (s)", "Total Tokens",
                                          "Tokens/s"], tablefmt='grid'))
                f.write("\n\n")
        f.write("--- Global Benchmark Results Summary ---\n\n")
        global_table = []
        for idx, _, _, res in results:
            table, headers = generate_stats_table(res, interactive=False, bench_id=idx if len(results) > 1 else None)
            for row, _ in table:
                global_table.append(row)
        f.write(tabulate(global_table, headers=headers, tablefmt='grid'))
        f.write("\n==================================================\n")
    print(f"\n{'All benchmarks' if args.file else 'Benchmark'} completed. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run benchmarks")
    parser.add_argument('-f', '--file', type=str, help="Config JSON file for non-interactive mode")
    parser.add_argument('-s', '--server', action='store_true', help="Run in server mode with FastAPI")
    args = parser.parse_args()

    if args.server:
        run_server_mode()
    else:
        results, summaries = run_non_interactive_mode(args.file) if args.file else run_interactive_mode()