import argparse
from server import run_server_mode
from benchmark import run_interactive_mode, run_non_interactive_mode

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks")
    parser.add_argument('-f', '--file', type=str, help="Config JSON file for non-interactive mode")
    parser.add_argument('-s', '--server', action='store_true', help="Run in server mode with FastAPI")
    args = parser.parse_args()

    if args.server:
        run_server_mode()
    else:
        results, summaries = run_non_interactive_mode(args.file) if args.file else run_interactive_mode()
        # Save results logic remains the same (omitted for brevity, can be moved to benchmark.py)

if __name__ == "__main__":
    main()