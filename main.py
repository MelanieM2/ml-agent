# main.py — command-line entry point for running one full ml-agent
# session. Real, committed replacement for run_smoke_test.py's role as
# "the way to actually start a session" -- run_smoke_test.py stays
# gitignored/hardcoded, for manual debugging only.
#
# Everything below just asks a real person the two genuine judgment
# calls run_session needs (dataset_name, optimization_target -- see
# agent.py's _format_initial_context docstring for why those two,
# specifically, are never silently decided on the caller's behalf),
# then calls run_session exactly the way run_smoke_test.py already
# does. No new orchestration logic lives here.

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # must run before genai.Client() is constructed inside run_session

from ml_agent.agent import run_session
from ml_agent.dataset import DATASET_LOADERS
from ml_agent.gemini_client import DEFAULT_MODEL, MAX_ITERATIONS, format_log

# The exact four metrics evaluate_model computes and returns (trainer.py)
# -- optimization_target is constrained to this set so Gemini is never
# pointed at a metric the tool results can't actually support. Closes a
# gap flagged as early as TECHNICAL_NOTES.md Part 2 (§2.3): optimization_
# target was previously unvalidated free text.
VALID_TARGETS = ["recall", "precision", "accuracy", "f1"]


def _prompt_for_dataset() -> str:
    """Interactive fallback when --dataset is omitted. Lists the real
    DATASET_LOADERS registry (dataset.py) rather than a hardcoded menu,
    so a new dataset added there later appears here automatically."""
    names = list(DATASET_LOADERS.keys())
    print("Available datasets:")
    for i, name in enumerate(names, start=1):
        print(f"  {i}. {name} — {DATASET_LOADERS[name].description}")

    while True:
        choice = input(f"Choose a dataset [{'/'.join(names)}]: ").strip()
        if choice in DATASET_LOADERS:
            return choice
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        print(f"Unrecognized dataset {choice!r}. Valid options: {', '.join(names)}")


def _prompt_for_target() -> str:
    """Interactive fallback when --target is omitted. Constrained to
    VALID_TARGETS -- see module docstring above for why. Empty input
    (just pressing Enter) defaults to 'recall'."""
    choice = input(
        f"Optimize for [{'/'.join(VALID_TARGETS)}] (default: recall): "
    ).strip().lower()
    if not choice:
        return "recall"
    while choice not in VALID_TARGETS:
        choice = input(
            f"Unrecognized target {choice!r}. Choose one of "
            f"[{'/'.join(VALID_TARGETS)}]: "
        ).strip().lower()
    return choice


def parse_args() -> argparse.Namespace:
    """CLI flags for everything run_session accepts. --dataset/--target
    fall back to an interactive prompt when omitted (the two genuine
    judgment calls); everything else silently uses run_session's own
    defaults when omitted, with no prompt -- these are secondary/
    advanced knobs that shouldn't interrupt a normal run."""
    parser = argparse.ArgumentParser(
        description=(
            "Run one full ml-agent session: Gemini proposes, trains, and "
            "evaluates scikit-learn models against a chosen dataset, using "
            "the given optimization target."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_LOADERS.keys()),
        default=None,
        help="Dataset to use. Prompted interactively if omitted.",
    )
    parser.add_argument(
        "--target",
        choices=VALID_TARGETS,
        default=None,
        help=(
            "Metric to optimize for. Prompted interactively if omitted "
            "(default on empty input: recall)."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_ITERATIONS,
        help=f"Max agent-loop iterations before giving up (default: {MAX_ITERATIONS}).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test split (default: 42).",
    )
    # BooleanOptionalAction gives both --log-iterations and
    # --no-log-iterations for free, from one flag definition.
    parser.add_argument(
        "--log-iterations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Print and save the full step-by-step tool-call trail "
            "(default: on, unlike run_session's own library default of "
            "off -- a first-time user benefits from seeing the agent's "
            "full reasoning). Use --no-log-iterations for a quieter run "
            "showing only the final result."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Printed unconditionally, at the very start of every run -- not
    # just when --dataset is omitted -- so it's visible to a contributor
    # even when running with flags fully specified. Points at
    # TECHNICAL_NOTES.md Part 5, §5.8 (added this session specifically
    # so this reference has somewhere real to point to).
    print(
        "(Want to add a new dataset? Binary classification only for now "
        "-- see TECHNICAL_NOTES.md, Part 5, §5.8 'Adding a new dataset' "
        "for the 3-step process.)\n"
    )

    dataset_name = args.dataset or _prompt_for_dataset()
    optimization_target = args.target or _prompt_for_target()

    config = {
        "dataset": dataset_name,
        "target": optimization_target,
        "model": args.model,
        "max_iterations": args.max_iterations,
        "random_state": args.random_state,
        "log_iterations": args.log_iterations,
    }

    # Effective-config summary, printed once, never asked -- so every
    # option is discoverable without ever pausing the run to ask about it.
    print(
        "\nRunning: "
        f"dataset={config['dataset']}, target={config['target']}, "
        f"model={config['model']}, max_iterations={config['max_iterations']}, "
        f"random_state={config['random_state']}, "
        f"log_iterations={config['log_iterations']}"
    )
    print(
        "(override any of these next time with --dataset, --target, "
        "--model, --max-iterations, --random-state, --no-log-iterations)\n"
    )

    start_time = time.time()
    result = run_session(
        dataset_name,
        optimization_target,
        random_state=args.random_state,
        model=args.model,
        max_iterations=args.max_iterations,
        log_iterations=args.log_iterations,
    )
    elapsed_seconds = time.time() - start_time

    print(result["status"], "-", result["iterations"], "iterations")
    if elapsed_seconds < 60:
        print(f"total elapsed time: {elapsed_seconds:.1f}s")
    else:
        minutes, seconds = divmod(elapsed_seconds, 60)
        print(f"total elapsed time: {int(minutes)}m {seconds:.1f}s")

    if "log" in result:
        print()
        print(format_log(result["log"]))

    # Persisted unconditionally, unlike run_smoke_test.py (which only
    # writes when LOG_ITERATIONS=True) -- a --no-log-iterations run's
    # final result is still worth keeping. Same smoke_test_log_<ts>.json
    # naming convention as run_smoke_test.py, deliberately, so both entry
    # points feed the same results/ directory compare_runs.py already
    # scans -- flagged for Melanie's confirmation once compare_runs.py's
    # real current content is checked, in case its scan logic is more
    # specific than "everything in results/".
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    log_path = Path(f"results/smoke_test_log_{timestamp}.json")
    log_path.parent.mkdir(exist_ok=True, parents=True)

    persisted = {**result, "elapsed_seconds": elapsed_seconds, "config": config}
    log_path.write_text(json.dumps(persisted, indent=2))
    print(f"\n(full result also written to {log_path}, elapsed {elapsed_seconds:.1f}s)")


if __name__ == "__main__":
    main()