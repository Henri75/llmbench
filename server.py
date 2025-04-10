import os
import logging
import asyncio
import signal
import sys
import json
from datetime import datetime
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from config import STATIC_DIR, CONFIG_DIR, EXPORT_DIR, progress_state  # Import progress_state from config
from benchmark import (run_non_interactive_mode_from_config, get_available_servers, fetch_models,
                      find_common_models, generate_stats_table, select_servers)
from utils import load_app_config, save_app_config, get_display_name, extract_short_name, extract_model_version, extract_model_size, extract_quantization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Benchmark Server")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class BenchmarkConfig(BaseModel):
    config: dict

class ServerConfig(BaseModel):
    name: str
    base_url: str
    label: str
    model: str = None

DEFAULT_SERVERS = load_app_config()["servers"] or {}

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
        servers = config["servers"] or {}
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
        servers = config["servers"] or {}
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
    DEFAULT_SERVERS = config["servers"] or {}
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
            from tabulate import tabulate
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
        content = f.read().strip()
        if not content:
            raise HTTPException(status_code=400, detail="Config file is empty")
        try:
            config = json.loads(content)
            servers = DEFAULT_SERVERS.copy()
            if config.get("servers") is not None:
                servers.update(config["servers"])
            total_steps = 0
            for bench in config["benchmarks"]:
                bench_servers = servers.copy()
                if bench.get("servers") is not None:
                    bench_servers.update(bench["servers"])
                available = get_available_servers(bench_servers)
                num_servers = bench.get("num_servers", len(available))
                num_repeats = max(bench.get("num_repeats", 1), 1)
                total_steps += num_servers * num_repeats
            progress_state.update({"current": 0, "total": total_steps, "message": "Config loaded", "config": config})
            return config
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in config file: {str(e)}")

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

def run_server_mode():
    print("Starting LLM Benchmark Server on http://localhost:8086")
    def shutdown(sig, frame):
        print("\nReceived shutdown signal. Exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8086)