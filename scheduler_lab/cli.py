import argparse
import json
from pathlib import Path
import sys

from .experiment import calibrate, load_config, make_plan, run_matrix
from .linux import parse_cpus


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def nonnegative(value):
    number = float(value)
    if not 0 <= number < float("inf"):
        raise argparse.ArgumentTypeError("Must be finite and nonnegative")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="Linux scheduling experiments for CPU-only Ollama inference")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Print the randomized matrix without starting workloads")
    plan.add_argument("--config", default="configs/proposal.json")
    plan.add_argument("--cpus", default="1-3")
    calibrate_parser = commands.add_parser("calibrate", help="Create an exact 128-token prompt lock using an existing daemon")
    calibrate_parser.add_argument("--url", default="http://127.0.0.1:11434")
    calibrate_parser.add_argument("--model", default="llama3.2:3b-instruct-q4_K_M")
    calibrate_parser.add_argument("--prompt", default="prompts/base.txt")
    calibrate_parser.add_argument("--prompt-tokens", type=positive_int, default=128)
    calibrate_parser.add_argument("--threads", type=positive_int, default=3)
    calibrate_parser.add_argument("--timeout", type=positive_int, default=180)
    calibrate_parser.add_argument("--output", default="prompts/locked.json")
    matrix = commands.add_parser("matrix", help="Run real experiments on a private daemon per trial")
    matrix.add_argument("--config", default="configs/proposal.json")
    matrix.add_argument("--cpus", default="1-3")
    matrix.add_argument("--models-dir", required=True)
    matrix.add_argument("--prompt-lock", default="prompts/locked.json")
    matrix.add_argument("--output", required=True)
    matrix.add_argument("--tokens", type=positive_int, default=128)
    matrix.add_argument("--timeout", type=positive_int, default=300, help="Total per-trial deadline, including startup and warmup")
    matrix.add_argument("--settle", type=nonnegative, default=2)
    matrix.add_argument("--cooldown", type=nonnegative, default=2)
    matrix.add_argument("--keep-going", action="store_true", help="Retain failures and continue; never substitute another policy")
    analyze = commands.add_parser("analyze", help="Summarize completed trials and draw four figures")
    analyze.add_argument("input")
    analyze.add_argument("--output", required=True)
    demo = commands.add_parser("demo", help="Generate clearly labeled synthetic fixtures without Ollama or Linux")
    demo.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            print(json.dumps(make_plan(load_config(args.config), len(parse_cpus(args.cpus))), indent=2))
        elif args.command == "calibrate":
            lock = calibrate(args)
            print(f"Locked {lock['prompt_eval_count']} input tokens, sha256={lock['sha256']}")
        elif args.command == "matrix":
            args.cpu_set = parse_cpus(args.cpus)
            return_code = run_matrix(args)
            if return_code:
                raise SystemExit(return_code)
        elif args.command == "analyze":
            from .analysis import analyze_results
            analyze_results(Path(args.input), Path(args.output))
        elif args.command == "demo":
            from .demo import create_demo
            create_demo(Path(args.output))
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
