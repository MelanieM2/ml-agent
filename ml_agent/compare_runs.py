# ml_agent/compare_runs.py — turns several results/result_log_*.json
# files into one comparison file, so multiple agent runs can be looked
# at side by side instead of one file at a time.
#
# Added 2026-07-24, part of item 2 on the running TODO list -- the
# ORIGINAL motivating question ("why did one run converge faster with
# no ConvergenceWarning than another") needed multiple runs' logs
# persisted and compared side by side, which wasn't possible before
# that session's timestamped-filename change to run_smoke_test.py
# (previously results/smoke_test_log.json was overwritten every run,
# so only ever one run's data existed at a time).
#
# Filename prefix updated 2026-07-29: smoke_test_log_<timestamp>.json
# -> result_log_<timestamp>_<dataset_name>.json (confirmed with
# Melanie). glob pattern below updated to match. See rename_results.py
# for the one-time migration of pre-existing files in results/.

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def summarize_run(run_data: dict[str, Any], source_file: str | None = None) -> dict[str, Any]:
    """Turns one already-loaded run dict (the full contents of one
    results/result_log_*.json file) into one flat, comparable summary
    row.

    Pure function -- no file I/O, no live API call -- same "isolate the
    checkable part" instinct as validate_split (agent.py) and
    validate_hyperparameters (trainer.py). Testable with a hand-built
    dict, no real run needed.

    DEFENSIVE ON PURPOSE: this project's results/ folder now contains a
    genuine mix of file vintages -- runs from before the 2026-07-24
    session don't have an "elapsed_seconds" key at the top level at
    all, and their train_model log entries don't have a "warnings" key
    in their result dict at all (that field didn't exist before that
    session's trainer.py change). Every access below uses .get(...)
    with an explicit default rather than direct indexing, specifically
    so this function doesn't crash on an older file -- it should
    degrade to None/empty rather than raise. This is unrelated to, and
    unaffected by, the 2026-07-29 filename-prefix rename -- it's about
    what's inside the file, not what the file is called.

    Returns a dict with these keys, always present (None or [] when the
    source run doesn't have the underlying data):
      source_file, status, iterations, elapsed_seconds, model_sequence,
      final_model_type, final_metrics, warnings_encountered,
      convergence_reasoning
    """
    log: list[dict[str, Any]] = run_data.get("log", [])
    # NOTE: if log_iterations was False for a given run, "log" won't be
    # in run_data at all -- .get(..., []) means everything below simply
    # comes back empty/None for that run, rather than raising. A run
    # summarized this way is still listed (status/iterations/
    # elapsed_seconds are still real), just without any per-model detail.

    model_sequence: list[str] = []
    warnings_encountered: list[dict[str, Any]] = []
    final_model_type: str | None = None
    final_metrics: dict[str, Any] | None = None
    convergence_reasoning: str | None = None

    for entry in log:
        tool_name = entry.get("tool_name")
        tool_args = entry.get("tool_args") or {}
        result = entry.get("result") or {}

        if tool_name == "train_model":
            model_type = tool_args.get("model_type")
            if model_type is not None:
                model_sequence.append(model_type)

            # .get("warnings", []) -- NOT result["warnings"] -- see
            # docstring: older files never had this key at all.
            for w in result.get("warnings", []):
                warnings_encountered.append({
                    "iteration": entry.get("iteration"),
                    "model_type": model_type,
                    "category": w.get("category"),
                    "message": w.get("message"),
                })

        elif tool_name == "evaluate_model":
            # Judgment call, flagged: takes the LAST evaluate_model seen
            # in the log as "the final model" -- true for every run
            # observed so far (Gemini always evaluates the model it's
            # about to accept last), but not something this function
            # verifies against record_convergence_decision's own
            # reasoning text. If a run ever evaluates a model, rejects
            # it, then stops WITHOUT evaluating anything further, this
            # would misreport that rejected model as "final". Not
            # observed in any real run yet -- flagged for Melanie to
            # confirm this assumption still holds if it comes up.
            final_model_type = result.get("model_type")
            final_metrics = {
                "accuracy": result.get("accuracy"),
                "precision": result.get("precision"),
                "recall": result.get("recall"),
                "f1": result.get("f1"),
                "confusion_matrix": result.get("confusion_matrix"),
            }

        elif tool_name == "record_convergence_decision":
            # Last one wins if there are several (a run can call this
            # with continue_iterating=True more than once before
            # finally stopping) -- deliberately not filtered to only
            # the stopping call, so a run that hit max_iterations while
            # Gemini was still mid-reasoning still shows its last stated
            # reasoning here, not None.
            convergence_reasoning = result.get("reasoning")

    return {
        "source_file": source_file,
        "status": run_data.get("status"),
        "iterations": run_data.get("iterations"),
        "elapsed_seconds": run_data.get("elapsed_seconds"),  # None on older files
        "model_sequence": model_sequence,
        "final_model_type": final_model_type,
        "final_metrics": final_metrics,
        "warnings_encountered": warnings_encountered,
        "convergence_reasoning": convergence_reasoning,
    }


def build_comparison(
    results_dir: Path = Path("results"),
    dataset_name: str | None = None,
) -> list[dict[str, Any]]:
    """Scans results_dir for result_log_*.json files and returns one
    summarize_run() row per file.

    dataset_name, when given, restricts the scan to
    result_log_*_<dataset_name>.json only -- confirmed with Melanie
    2026-07-29: comparing runs across different datasets is never
    allowed, since their metrics (accuracy/precision/recall/f1) aren't
    comparable against different data. main.py's `compare` subcommand
    always passes an explicit, validated dataset_name and never calls
    this with None.

    dataset_name=None (scan everything regardless of dataset) is kept
    only for this module's own standalone `python -m
    ml_agent.compare_runs` entry point, preserving its original
    behavior -- flagged for Melanie to confirm whether that standalone
    entry point should also be restricted to one dataset at a time, now
    that main.py's `compare` subcommand is the primary, documented way
    to do this.

    Not capped at any particular number of runs -- picks up however
    many matching files currently exist in results_dir, whether that's
    2 or 20. Sorted by filename, which still sorts chronologically
    as-is, since the naming convention is
    result_log_<YYYY_MM_DD_HHMMSS>_<dataset_name>.json -- the timestamp
    segment immediately follows the fixed prefix, so a whole-filename
    string sort compares timestamps first and only falls back to
    dataset_name as a tiebreaker on the (currently impossible, given
    HHMMSS granularity) case of two runs sharing the exact same second.
    This was deliberately checked when the naming convention changed
    2026-07-29 -- putting dataset_name BEFORE the timestamp instead
    would have broken this sort (all of one dataset's runs would sort
    before all of another's, regardless of when they actually ran).
    """
    pattern = f"result_log_*_{dataset_name}.json" if dataset_name else "result_log_*.json"
    rows = []
    for path in sorted(results_dir.glob(pattern)):
        run_data = json.loads(path.read_text())
        rows.append(summarize_run(run_data, source_file=path.name))
    return rows


if __name__ == "__main__":
    rows = build_comparison()

    timestamp = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")
    out_path = Path(f"results/comparison_{timestamp}.json")
    out_path.write_text(json.dumps(rows, indent=2))

    print(f"Compared {len(rows)} run(s) -> {out_path}")