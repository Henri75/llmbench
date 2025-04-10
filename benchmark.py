from tabulate import tabulate
from datetime import datetime
import time
import statistics
import json
import requests
import sys
import os
from openai import OpenAI
from utils import (load_app_config, get_display_name, check_server_availability, fetch_models,
                  extract_short_name, extract_model_version, extract_model_size, extract_quantization)
from config import progress_state  # Import global progress_state from config

DEFAULT_SERVERS = load_app_config()["servers"] or {}

def benchmark(config, messages, server_name, repeat, total_repeats):
    progress_state["message"] = f"Benchmarking {get_display_name(server_name, config)} - Repeat {repeat}/{total_repeats}"
    print(f"\n{get_display_name(server_name, config)} Server - Repeat {repeat}/{total_repeats} - Model: {config['model']}")
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

def get_available_servers(server_configs):
    if server_configs is None:
        return {}
    return {name: config for name, config in server_configs.items() if check_server_availability(name, config)}

def select_servers(available_servers, num_servers, interactive=False, combo=None):
    from itertools import combinations
    if not available_servers:
        raise ValueError("No available servers provided")
    server_list = list(available_servers.keys())
    combos = ['+'.join(combo) for combo in combinations(server_list, num_servers)] if num_servers <= len(server_list) else []
    if not combos:
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
                row.extend([model['name'] if server_name == s else '-', model['timestamp'] if server_name == s else '-'])
            table_data.append(row)
    headers = ['Server', 'Model Name', 'Model Version', 'Model Size', 'Quantization', 'Instruct',
               'Server1 Model', 'Server1 Created', 'Server2 Model', 'Server2 Modified', 'Server3 Model', 'Server3 Created']
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

def run_benchmarks(selected_servers, messages, num_repeats, base_steps=0):
    total_steps_per_benchmark = len(selected_servers) * num_repeats
    results = {}
    step_increment = 0
    for name, config in selected_servers:
        outputs, total_times, ttfts, gen_times, tokens, tokens_per_sec = [], [], [], [], [], []
        print(f"\nStarting benchmark for {get_display_name(name, config)} server...")
        for i in range(num_repeats):
            step_increment += 1
            progress_state["current"] = base_steps + step_increment
            print(f"Progress: {progress_state['current']}/{progress_state['total']} - {progress_state['message']}")
            out, tt, ttf, gt, tok, tps = benchmark(config, messages, name, i + 1, num_repeats)
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
        stats = lambda x: (min(x) if x else 0, max(x) if x else 0, sum(x) / len(x) if x else 0, statistics.stdev(x) if len(x) > 1 else 0)
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
        row.extend([f"{v:.2f}" for stat in (tt_stats, ttf_stats, gt_stats, tps_stats) for v in (stat[:3] if interactive else stat)])
        table.append((row, last_output))
    headers = ["Server", "Full Model Name"] if interactive else ["Benchmark", "Server", "Full Model Name", "Model Name", "Version", "Size", "Quantization"]
    headers.extend(sum([["Min Total (s)", "Max Total (s)", "Avg Total (s)"] + (["Std Total (s)"] if not interactive else []),
                        ["Min TTFT (s)", "Max TTFT (s)", "Avg TTFT (s)"] + (["Std TTFT (s)"] if not interactive else []),
                        ["Min Gen (s)", "Max Gen (s)", "Avg Gen (s)"] + (["Std Gen (s)"] if not interactive else []),
                        ["Min Tokens/s", "Max Tokens/s", "Avg Tokens/s"] + (["Std Tokens/s"] if not interactive else [])], []))
    return table, headers

def run_interactive_mode():
    servers = get_available_servers(DEFAULT_SERVERS.copy())
    if not servers:
        print("No servers available.")
        sys.exit(1)
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
    progress_state.update({"current": 0, "total": len(selected_servers) * num_repeats, "message": ""})
    results, _ = run_benchmarks(selected_servers, messages, num_repeats)

    summary = [{"Benchmark": "Interactive", "Servers": combo_display,
                "Model": ", ".join(f"{get_display_name(n, c)}: {c['model']}" for n, c in selected_servers),
                "Query": messages[0]['content'], "Repeats": num_repeats}]
    print("\n--- Benchmark Queries Summary ---\n" + tabulate(
        [[d[h] for h in "Benchmark Servers Model Query Repeats".split()] for d in summary],
        headers="Benchmark Servers Model Query Repeats".split(), tablefmt='grid'))
    print("\n--- Benchmark Results ---")
    table, headers = generate_stats_table(results)
    print(tabulate([row for row, _ in table], headers=headers, tablefmt='grid'))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_{timestamp}.txt"
    with open(output_file, 'w') as f:
        f.write("==================================================\n")
        f.write("          LLM Benchmark Report\n")
        f.write(f"          Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("==================================================\n\n")
        f.write("--- Benchmark Parameters Summary ---\n\n")
        params_table = [[summary[0]["Benchmark"], len(selected_servers), summary[0]["Model"], summary[0]["Repeats"], summary[0]["Query"]]]
        f.write(tabulate(params_table, headers=["Benchmark", "Number of Servers", "Models", "Number of Iterations", "LLM Query"], tablefmt='grid'))
        f.write("\n\n")
        f.write("--- Servers and Models Details ---\n\n")
        servers_table = [["Interactive", get_display_name(n, c), c["base_url"], extract_short_name(c["model"]),
                          extract_model_version(c["model"]), extract_model_size(c["model"]), extract_quantization(c["model"]), c["model"]]
                         for n, c in selected_servers]
        f.write(tabulate(servers_table, headers=["Benchmark", "Server Label", "Base URL", "Model Short Name", "Model Version", "Model Size", "Quantization", "Full Model Name"], tablefmt='grid'))
        f.write("\n\n")
        f.write("--- Detailed Benchmark Results ---\n\n")
        for name, data in results.items():
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
            f.write(tabulate(pass_table, headers=["Pass", "Total Time (s)", "TTFT (s)", "Generation Time (s)", "Total Tokens", "Tokens/s"], tablefmt='grid'))
            f.write("\n\n")
        f.write("--- Global Benchmark Results Summary ---\n\n")
        table, headers = generate_stats_table(results, interactive=False)
        f.write(tabulate([row for row, _ in table], headers=headers, tablefmt='grid'))
        f.write("\n==================================================\n")
    print(f"\nBenchmark completed. Results saved to {output_file}")
    return [(1, {}, selected_servers, results)], summary

def run_non_interactive_mode(config_file):
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' does not exist.")
        sys.exit(1)
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{config_file}': {str(e)}")
        sys.exit(1)

    servers = DEFAULT_SERVERS.copy() if DEFAULT_SERVERS is not None else {}
    if config.get("servers") is not None:
        servers.update(config["servers"])

    progress_state.update({"current": 0, "total": 0, "message": ""})
    total_steps = sum(len(get_available_servers(dict(servers).update(b.get("servers", {}) or {}))) * b.get("num_repeats", 1) for b in config["benchmarks"])
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
        results, steps = run_benchmarks(selected_servers, messages, num_repeats, base_steps)
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_{timestamp}.txt"
    with open(output_file, 'w') as f:
        f.write("==================================================\n")
        f.write("          LLM Benchmark Report\n")
        f.write(f"          Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("==================================================\n\n")
        f.write("--- Benchmark Parameters Summary ---\n\n")
        params_table = []
        for r in results_list:
            idx, bench, selected_servers, _ = r
            models_str = "Common Model: " + bench.get("model_name", "N/A") if "model_name" in bench else ", ".join(
                f"{get_display_name(n, c)}: {c['model']}" for n, c in selected_servers)
            params_table.append([f"Benchmark {idx}", len(selected_servers), models_str, bench.get("num_repeats", 1), summaries[0]["Query"]])
        f.write(tabulate(params_table, headers=["Benchmark", "Number of Servers", "Models", "Number of Iterations", "LLM Query"], tablefmt='grid'))
        f.write("\n\n")
        f.write("--- Servers and Models Details ---\n\n")
        servers_table = []
        for idx, _, selected_servers, _ in results_list:
            for name, config in selected_servers:
                servers_table.append([f"Benchmark {idx}", get_display_name(name, config), config["base_url"],
                                      extract_short_name(config["model"]), extract_model_version(config["model"]),
                                      extract_model_size(config["model"]), extract_quantization(config["model"]), config["model"]])
        f.write(tabulate(servers_table, headers=["Benchmark", "Server Label", "Base URL", "Model Short Name", "Model Version", "Model Size", "Quantization", "Full Model Name"], tablefmt='grid'))
        f.write("\n\n")
        f.write("--- Detailed Benchmark Results ---\n\n")
        for idx, _, selected_servers, res in results_list:
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
                f.write(tabulate(pass_table, headers=["Pass", "Total Time (s)", "TTFT (s)", "Generation Time (s)", "Total Tokens", "Tokens/s"], tablefmt='grid'))
                f.write("\n\n")
        f.write("--- Global Benchmark Results Summary ---\n\n")
        global_table = []
        for idx, _, _, res in results_list:
            table, headers = generate_stats_table(res, interactive=False, bench_id=idx)
            for row, _ in table:
                global_table.append(row)
        f.write(tabulate(global_table, headers=headers, tablefmt='grid'))
        f.write("\n==================================================\n")
    print(f"\nAll benchmarks completed. Results saved to {output_file}")
    return results_list, summaries

def run_non_interactive_mode_from_config(config_data):
    servers = DEFAULT_SERVERS.copy() if DEFAULT_SERVERS is not None else {}
    if config_data.get("servers") is not None:
        servers.update(config_data["servers"])
    results_list, summaries = [], []
    base_steps = 0
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
        print(f"Progress: current={progress_state['current']}, total={progress_state['total']}, message={progress_state['message']}")
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
        print(f"Progress: current={progress_state['current']}, total={progress_state['total']}, message={progress_state['message']}")
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
        print(f"Progress: current={progress_state['current']}, total={progress_state['total']}, message={progress_state['message']}")
        results, steps = run_benchmarks(selected_servers, messages, num_repeats, base_steps)
        base_steps += steps
        results_list.append((idx, bench, selected_servers, results))
        summaries.append({"Benchmark": f"Benchmark {idx}", "Servers": combo_display,
                          "Model": ", ".join(f"{get_display_name(n, c)}: {c['model']}" for n, c in selected_servers),
                          "Query": messages[0]["content"], "Repeats": num_repeats})
    return results_list, summaries