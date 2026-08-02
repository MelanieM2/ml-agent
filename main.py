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
#
# `export` and `report` subcommands added 2026-07-31 -- both thin
# wrappers around reporting.py, same "no new logic in main.py" principle
# already used for `compare` below.

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # must run before genai.Client() is constructed inside run_session

from ml_agent.agent import run_session
from ml_agent.compare_runs import build_comparison
from ml_agent.dataset import DATASET_LOADERS
from ml_agent.gemini_client import DEFAULT_MODEL, MAX_ITERATIONS, format_log
from ml_agent.reporting import load_rows, to_csv, to_markdown

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI flags for everything run_session accepts. --dataset/--target
    fall back to an interactive prompt when omitted (the two genuine
    judgment calls); everything else silently uses run_session's own
    defaults when omitted, with no prompt -- these are secondary/
    advanced knobs that shouldn't interrupt a normal run.

    Accepts an explicit argv so main() can hand this a pre-sliced list
    (e.g. with a leading "run" token already stripped) rather than
    always reading sys.argv directly -- see main()'s subcommand
    handling below."""
    parser = argparse.ArgumentParser(
        description=(
            "Run one full ml-agent session: Gemini proposes, trains, and "
            "evaluates scikit-learn models against a chosen dataset, using "
            "the given optimization target. "
            "(Also see the separate 'compare', 'export', and 'report' "
            "subcommands: `python main.py compare` compares all persisted "
            "past runs; `python main.py export` writes them to CSV; "
            "`python main.py report <file>` renders a Markdown viewer for "
            "one run or one comparison file -- see each subcommand's own "
            "--help. A lower-level standalone entry point, "
            "`python -m ml_agent.compare_runs <dataset>`, also exists for "
            "ad-hoc comparisons without going through this CLI -- same "
            "one-dataset-at-a-time rule applies.)"
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
    return parser.parse_args(argv)


def _run_compare(argv: list[str]) -> None:
    """Handles `python main.py compare [--dataset NAME] [--results-dir DIR]`.

    Thin wrapper directly reusing compare_runs.build_comparison() --
    no new comparison logic lives here, same "no new orchestration
    logic in main.py" principle as the run path above.

    --dataset is REQUIRED, not just recommended -- confirmed with
    Melanie 2026-07-29: comparing runs across different datasets would
    combine incomparable metrics into one file, so this is never
    allowed, not even as an opt-in "compare everything" mode. If
    omitted, or given but not a real dataset (typo), this prints a
    warning explaining why and falls back to the SAME interactive
    picker _prompt_for_dataset() already uses for the `run` path above
    -- one piece of picker logic, not two -- rather than argparse's
    own choices= mechanism, which would just print a bare usage error
    and exit instead of explaining itself and re-prompting.

    Writes results/comparison_<timestamp>_<dataset>.json -- the
    dataset name in the filename itself is what actually solves the
    original problem Melanie flagged (comparison_<timestamp>.json gave
    no indication of which dataset a comparison covered)."""
    parser = argparse.ArgumentParser(
        prog="main.py compare",
        description=(
            "Compare all persisted results/result_log_*.json runs for ONE "
            "dataset into one results/comparison_<timestamp>_<dataset>.json "
            "summary file. --dataset is required -- comparing runs across "
            "different datasets would mix metrics that aren't comparable "
            "against different data, so this is never allowed. If omitted "
            "or misspelled, you'll be prompted to choose interactively."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Dataset to compare runs for. Required -- if omitted or not a "
            f"real dataset name ({', '.join(DATASET_LOADERS.keys())}), "
            "you'll be prompted to choose one interactively."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help=(
            "Directory to scan for result_log_*.json files. Only needed if "
            "your runs live somewhere other than the default results/ folder."
        ),
    )
    args = parser.parse_args(argv)

    dataset_name = args.dataset
    if dataset_name is None:
        print(
            "A dataset name is required to compare runs -- mixing datasets "
            "would combine metrics that aren't comparable against different data."
        )
        dataset_name = _prompt_for_dataset()
    elif dataset_name not in DATASET_LOADERS:
        print(f"Unrecognized dataset {dataset_name!r}.")
        dataset_name = _prompt_for_dataset()

    rows = build_comparison(args.results_dir, dataset_name=dataset_name)

    if not rows:
        print(f"No result_log_*_{dataset_name}.json files found in {args.results_dir}/ yet -- nothing to compare.")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")
    out_path = args.results_dir / f"comparison_{timestamp}_{dataset_name}.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"Compared {len(rows)} run(s) for dataset={dataset_name!r} -> {out_path}")


def _run_export(argv: list[str]) -> None:
    """Handles `python main.py export --dataset NAME [--results-dir DIR] [--out PATH]`.

    CSV spreadsheet export -- TODO #10, Melanie's explicitly named gate
    before making the repo public. Mirrors `compare`'s exact pattern:
    --dataset is REQUIRED for the same reason build_comparison() already
    enforces it (mixing datasets' metrics into one file is never
    allowed) -- reuses that same required-dataset + interactive-picker-
    on-omission logic, not a second copy of it.

    Deliberately calls build_comparison() directly (compare_runs.py)
    rather than requiring an existing comparison_*.json file on disk
    first -- an export is "give me every run for this dataset as a
    spreadsheet," which is exactly what build_comparison() already
    gathers in memory; writing an intermediate JSON file first would be
    a pointless extra step. This also means an export always reflects
    every field summarize_run() currently produces (hyperparameters,
    dataset/target/random_state included), since it's not reading an
    older, possibly-stale comparison_*.json file.
    """
    parser = argparse.ArgumentParser(
        prog="main.py export",
        description=(
            "Export all persisted results/result_log_*.json runs for ONE "
            "dataset to a single CSV file, suitable for opening in Excel/"
            "Google Sheets. --dataset is required -- comparing/exporting "
            "runs across different datasets would mix metrics that aren't "
            "comparable against different data, so this is never allowed."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Dataset to export runs for. Required -- if omitted or not a "
            f"real dataset name ({', '.join(DATASET_LOADERS.keys())}), "
            "you'll be prompted to choose one interactively."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to scan for result_log_*.json files (default: results/).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: results/export_<timestamp>_<dataset>.csv).",
    )
    args = parser.parse_args(argv)

    dataset_name = args.dataset
    if dataset_name is None:
        print(
            "A dataset name is required to export runs -- mixing datasets "
            "would combine metrics that aren't comparable against different data."
        )
        dataset_name = _prompt_for_dataset()
    elif dataset_name not in DATASET_LOADERS:
        print(f"Unrecognized dataset {dataset_name!r}.")
        dataset_name = _prompt_for_dataset()

    rows = build_comparison(args.results_dir, dataset_name=dataset_name)

    if not rows:
        print(f"No result_log_*_{dataset_name}.json files found in {args.results_dir}/ yet -- nothing to export.")
        return

    out_path = args.out
    if out_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")
        out_path = args.results_dir / f"export_{timestamp}_{dataset_name}.csv"

    to_csv(rows, out_path)
    print(f"Exported {len(rows)} run(s) for dataset={dataset_name!r} -> {out_path}")


def _run_report(argv: list[str]) -> None:
    """Handles `python main.py report <path> [--out PATH]`.

    Human-friendly Markdown viewer -- TODO #4, design agreed 2026-07-24
    (Markdown output, auto-detect by filename), built 2026-07-31.

    <path> can be EITHER a result_log_*.json (one run) or a
    comparison_*.json (several runs already summarized) --
    reporting.py's load_rows()/to_markdown() auto-detect which, by
    filename prefix only (Option A, confirmed with Melanie 2026-07-31:
    no JSON-content-inspection fallback -- these filenames aren't
    expected to be renamed by hand).
    """
    parser = argparse.ArgumentParser(
        prog="main.py report",
        description=(
            "Render a result_log_*.json (one run) or comparison_*.json "
            "(several runs) as a human-friendly Markdown report."
        ),
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a result_log_*.json or comparison_*.json file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .md path (default: same name as input, .md extension).",
    )
    args = parser.parse_args(argv)

    if not args.path.exists():
        print(f"File not found: {args.path}")
        return

    try:
        rows, kind = load_rows(args.path)
    except ValueError as e:
        print(str(e))
        return
    markdown = to_markdown(rows, kind, source_file=args.path.name)

    out_path = args.out or args.path.with_suffix(".md")
    out_path.write_text(markdown)
    print(f"Rendered {kind} report for {args.path.name} -> {out_path}")


def main() -> None:
    # Subcommand sniffing, done BEFORE argparse sees anything -- this is
    # what makes "compare"/"export"/"report" siblings of the default run
    # path without requiring every existing invocation to be retyped
    # with a leading "run" (confirmed with Melanie 2026-07-29 for
    # "compare"; same pattern extended 2026-07-31 for "export"/"report"):
    #   python main.py compare ...   -> handled entirely by _run_compare,
    #                                    returns immediately.
    #   python main.py export ...    -> handled entirely by _run_export,
    #                                    returns immediately.
    #   python main.py report ...    -> handled entirely by _run_report,
    #                                    returns immediately.
    #   python main.py run ...       -> "run" token stripped, everything
    #                                    else below proceeds exactly as
    #                                    if "run" had never been typed.
    #   python main.py ...           -> (no subcommand) proceeds exactly
    #                                    as it always has.
    raw_argv = sys.argv[1:]
    if raw_argv and raw_argv[0] == "compare":
        _run_compare(raw_argv[1:])
        return
    if raw_argv and raw_argv[0] == "export":
        _run_export(raw_argv[1:])
        return
    if raw_argv and raw_argv[0] == "report":
        _run_report(raw_argv[1:])
        return
    if raw_argv and raw_argv[0] == "run":
        raw_argv = raw_argv[1:]

    args = parse_args(raw_argv)

    # Printed unconditionally, at the very start of every run -- not
    # just when --dataset is omitted -- so it's visible to a contributor
    # even when running with flags fully specified. Points at
    # TECHNICAL_NOTES.md Part 5, §5.8 (added this session specifically
    # so this reference has somewhere real to point to).
    print(
        "(Want to add a new dataset? Binary classification only for now "
        "-- see TECHNICAL_NOTES.md, Part 5, §5.8 'Adding a new dataset' "
        "for the 3-step process.)\n"
        "(Note: the free tier for the Gemini API is rate-limited per "
        "minute. If you run this back-to-back several times quickly, "
        "you may hit a 429 RESOURCE_EXHAUSTED error -- just wait "
        "under a minute and try again.)\n"
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
    # final result is still worth keeping.
    #
    # Naming convention CHANGED 2026-07-29 (confirmed with Melanie):
    # smoke_test_log_<timestamp>.json -> result_log_<timestamp>_<dataset_name>.json
    #   - dataset_name deliberately comes AFTER the timestamp, not before,
    #     specifically so sorting by filename still sorts chronologically
    #     across every dataset (compare_runs.py's build_comparison()
    #     relies on exactly this) -- putting dataset_name first would have
    #     grouped all "breast_cancer" runs before all "climate" runs
    #     alphabetically, silently breaking that assumption.
    #   - compare_runs.py updated in the same session to glob
    #     "result_log_*.json" instead of the old "smoke_test_log_*.json"
    #     prefix -- see that file for the corresponding change.
    #   - existing files in results/ predating this change need a
    #     one-time migration; see rename_results.py.
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    log_path = Path(f"results/result_log_{timestamp}_{dataset_name}.json")
    log_path.parent.mkdir(exist_ok=True, parents=True)

    persisted = {**result, "elapsed_seconds": elapsed_seconds, "config": config}
    log_path.write_text(json.dumps(persisted, indent=2))
    print(f"\n(full result also written to {log_path}, elapsed {elapsed_seconds:.1f}s)")


if __name__ == "__main__":
    main()