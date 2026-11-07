"""Single supported command-line interface for the public release."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Sequence
from pathlib import Path

EXPERIMENTS = {
    "nominal": "nominal",
    "robustness": "robustness",
    "scaling-blocks": "scaling_blocks",
    "scaling-dimension": "scaling_dimension",
    "splitting": "splitting",
    "stress": "stress",
    "stress-sampling": "stress_sampling",
    "exponential-action": "exponential_action",
    "strength-scan": "strength_scan",
}


def main(arguments: Sequence[str] | None = None) -> None:
    """Dispatch documented commands; no command silently starts a long experiment."""
    parser = argparse.ArgumentParser(
        prog="stgrape", description="ST-GRAPE code and paper reproduction"
    )
    parser.add_argument("--version", action="version", version="ST-GRAPE 1.0.0")
    subs = parser.add_subparsers(dest="command")
    demo = subs.add_parser("demo", help="Run a two-qubit, two-iteration comparison")
    demo.add_argument("--output", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--iterations", type=int, default=2)
    reproduce = subs.add_parser(
        "reproduce", help="Regenerate paper figures/tables from archived data"
    )
    reproduce.add_argument("--output", type=Path, default=Path("outputs/paper"))
    reproduce.add_argument("--supplementary", action="store_true")
    subs.add_parser("verify-data", help="Verify every reference-data checksum and paper statistic")
    experiment = subs.add_parser(
        "experiment", help="Run an experiment with explicit arguments", add_help=False
    )
    experiment.add_argument("name", nargs="?", choices=tuple(EXPERIMENTS))
    experiment.add_argument("--config", type=Path, help="JSON preset containing an arguments list")
    args, remaining = parser.parse_known_args(arguments)
    if args.command != "experiment" and remaining:
        parser.error("unrecognized arguments: " + " ".join(remaining))
    if args.command == "demo":
        from .demo import run_demo

        run_demo(args.output, iterations=args.iterations)
    elif args.command == "verify-data":
        from .archive import verify_archive

        print(json.dumps(verify_archive(), indent=2))
    elif args.command == "reproduce":
        from .analysis import reproduce_paper

        reproduce_paper(args.output, supplementary=args.supplementary)
    elif args.command == "experiment":
        if args.name is None:
            experiment.print_help()
            return
        forwarded = []
        if args.config:
            config = json.loads(args.config.read_text(encoding="utf-8"))
            if config.get("experiment") != args.name:
                parser.error("Preset experiment does not match the requested command.")
            forwarded = config["arguments"]
            if not isinstance(forwarded, list) or not all(isinstance(v, str) for v in forwarded):
                parser.error("Preset arguments must be a list of strings.")
        module = importlib.import_module("stgrape.benchmarks." + EXPERIMENTS[args.name])
        module.main([*forwarded, *remaining])
    else:
        parser.print_help()
