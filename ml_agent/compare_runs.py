# ml_agent/compare_runs.py — turns several results/result_log_*.json
# files into one comparison file, so multiple agent runs can be looked
# at side by side instead of one file at a time.
#
# Each run file is scoped to a single dataset, encoded in its filename
# (result_log_<timestamp>_<dataset_name>.json). Comparing runs across
# different datasets is never allowed anywhere in this project's CLI
# surface, since accuracy/precision/recall/f1 aren't comparable across
# different data -- build_comparison enforces this via its optional
# dataset_name filter, and every real caller (main.py's compare/export
# subcommands, and this file's own standalone entry point below)
# always passes one explicitly.
#
# "Final model" resolution: rather than assuming the last evaluate_model
# call in a run's log is the model Gemini actually settled on, the final
# model is resolved structurally -- whichever evaluated model scores
# highest on the run's own optimization target. A run can legitimately
# evaluate one extra model purely as a comparison point after already
# deciding, so call order alone isn't a reliable signal. This is not
# text-parsing of Gemini's free-text convergence reasoning (considered
# and rejected as too fragile) -- it's a structural, metric-based
# resolution. Because it can still, rarely, disagree with what the
# reasoning text actually says, summarize_run() also flags
# final_model_ambiguous whenever the best-by-target pick differs from
# the last-evaluated pick -- a cheap, structural mismatch signal, not an
# attempt to parse or adjudicate the reasoning itself. A second live
# Gemini call to adjudicate that disagreement directly was considered
# and deliberately not built here -- that idea has since been folded
# into the broader human-in-the-loop design's post-hoc debug-assist mode
# (designed in some depth, not yet implemented), rather than being a
# one-off addition to this module.

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The same four metrics evaluate_model (trainer.py) computes -- kept
# here as a local constant rather than importing VALID_TARGETS from
# main.py, since compare_runs.py must stay import-independent of both
# main.py and reporting.py (see reporting.py's header comment for the
# full rationale: neither of those two should import from this module
# or from each other).
KNOWN_METRICS = {"accuracy", "precision", "recall", "f1"}


def summarize_run(run_data: dict[str, Any], source_file: str | None = None) -> dict[str, Any]:
    """Turns one already-loaded run dict (the full contents of one
    results/result_log_*.json file) into one flat, comparable summary
    row.

    Pure function -- no file I/O, no live API call -- same "isolate the
    checkable part" instinct as validate_split (agent.py) and
    validate_hyperparameters (trainer.py). Testable with a hand-built
    dict, no real run needed.

    Defensive by necessity: results/ contains a genuine mix of file
    vintages from before certain fields existed -- some runs have no
    top-level "elapsed_seconds" key, some train_model log entries have
    no "warnings" key in their result dict, and some runs predate the
    top-level "config" block entirely (no dataset/target/random_state
    ever recorded for them). Every access below uses .get(...) with an
    explicit default rather than direct indexing, so an older file
    degrades gracefully instead of raising.

    Returns a dict with these keys, always present (None/[]/empty when
    the source run doesn't have the underlying data):
      source_file, status, iterations, elapsed_seconds, dataset, target,
      random_state, model_sequence, final_model_type,
      final_hyperparameters, final_metrics, final_model_ambiguous,
      warnings_encountered, convergence_reasoning

    final_model resolution: every evaluate_model call this run is
    recorded, in order, as {model_ref, model_type, metrics}. If
    config["target"] is a known metric name, the "final" model is
    whichever evaluated model scores HIGHEST on that metric -- see the
    module header above for why call order alone isn't trusted. If
    target is missing or unrecognized (older files, from before it was
    persisted), falls back to "last evaluated" instead, since there's no
    target metric available to rank by.

    final_model_ambiguous: True when target IS known and the
    best-by-target pick differs from the last-evaluated pick -- a real
    disagreement worth a human's attention (see reporting.py for how
    this surfaces in output). False when target is known and the two
    picks agree. None when target is unknown -- there's no second
    opinion to compare against, so "ambiguous" isn't a meaningful
    question for those files.
    """
    log: list[dict[str, Any]] = run_data.get("log", [])
    config = run_data.get("config") or {}
    target = config.get("target")

    model_sequence: list[str] = []
    warnings_encountered: list[dict[str, Any]] = []
    convergence_reasoning: str | None = None

    # model_ref -> hyperparameters, built up as train_model calls are
    # seen -- lets us attach hyperparameters to whichever evaluation
    # ends up chosen as "final", regardless of which one that is.
    hyperparameters_by_ref: dict[str, dict[str, Any]] = {}

    # Every evaluate_model call this run, in encounter order -- NOT
    # collapsed to "the last one" anymore. This is what makes
    # best-by-target resolution possible.
    evaluations: list[dict[str, Any]] = []

    for entry in log:
        tool_name = entry.get("tool_name")
        tool_args = entry.get("tool_args") or {}
        result = entry.get("result") or {}

        if tool_name == "train_model":
            model_type = tool_args.get("model_type")
            if model_type is not None:
                model_sequence.append(model_type)

            model_ref = result.get("model_ref")
            if isinstance(model_ref, str):
                hyperparameters_by_ref[model_ref] = tool_args.get("hyperparameters") or {}

            for w in result.get("warnings", []):
                warnings_encountered.append({
                    "iteration": entry.get("iteration"),
                    "model_type": model_type,
                    "category": w.get("category"),
                    "message": w.get("message"),
                })

        elif tool_name == "evaluate_model":
            evaluations.append({
                "model_ref": result.get("model_ref"),
                "model_type": result.get("model_type"),
                "metrics": {
                    "accuracy": result.get("accuracy"),
                    "precision": result.get("precision"),
                    "recall": result.get("recall"),
                    "f1": result.get("f1"),
                    "confusion_matrix": result.get("confusion_matrix"),
                },
            })

        elif tool_name == "record_convergence_decision":
            # Last one wins if there are several -- a run can call this
            # with continue_iterating=True more than once before finally
            # stopping.
            convergence_reasoning = result.get("reasoning")

    # --- Resolve the "final" model -------------------------------------
    last_eval = evaluations[-1] if evaluations else None
    best_eval = last_eval
    final_model_ambiguous: bool | None = None

    if target in KNOWN_METRICS and evaluations:
        scored = [e for e in evaluations if e["metrics"].get(target) is not None]
        if scored:
            best_eval = max(scored, key=lambda e: e["metrics"][target])
            final_model_ambiguous = (
                last_eval is not None and best_eval["model_ref"] != last_eval["model_ref"]
            )
        # else: no evaluation has the target metric populated (shouldn't
        # happen given evaluate_model always computes all four, but
        # defensive against a malformed/older file) -- best_eval stays
        # last_eval, final_model_ambiguous stays None (nothing to
        # meaningfully compare).

    final_model_type = best_eval["model_type"] if best_eval else None
    final_metrics = best_eval["metrics"] if best_eval else None
    final_hyperparameters = (
        hyperparameters_by_ref.get(best_eval["model_ref"])
        if best_eval and isinstance(best_eval.get("model_ref"), str)
        else None
    )

    return {
        "source_file": source_file,
        "status": run_data.get("status"),
        "iterations": run_data.get("iterations"),
        "elapsed_seconds": run_data.get("elapsed_seconds"),
        "dataset": config.get("dataset"),
        "target": target,
        "random_state": config.get("random_state"),
        "model_sequence": model_sequence,
        "final_model_type": final_model_type,
        "final_hyperparameters": final_hyperparameters,
        "final_metrics": final_metrics,
        "final_model_ambiguous": final_model_ambiguous,
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
    result_log_*_<dataset_name>.json only. Comparing runs across
    different datasets is never allowed -- their metrics aren't
    comparable against different data. main.py's compare and export
    subcommands, and this file's own standalone entry point below,
    always pass an explicit dataset_name; dataset_name=None stays
    accepted here so this function itself remains a general-purpose,
    unopinionated utility rather than enforcing that restriction at this
    layer.

    Sorted by filename, which still sorts chronologically as-is: the
    naming convention is
    result_log_<YYYY_MM_DD_HHMMSS>_<dataset_name>.json, with the dataset
    name deliberately placed after the timestamp, not before -- putting
    it first would have made a whole-filename string sort group all of
    one dataset's runs together regardless of when they actually ran.
    """
    pattern = f"result_log_*_{dataset_name}.json" if dataset_name else "result_log_*.json"
    rows = []
    for path in sorted(results_dir.glob(pattern)):
        run_data = json.loads(path.read_text())
        rows.append(summarize_run(run_data, source_file=path.name))
    return rows


if __name__ == "__main__":
    import sys
    from ml_agent.dataset import DATASET_LOADERS

    def _prompt_for_dataset_standalone() -> str:
        """Minimal dataset picker for THIS standalone entry point only --
        deliberately not importing main.py's _prompt_for_dataset(): main.py
        already imports build_comparison from this module, so the reverse
        import would be circular. Kept intentionally simple (no retry-loop
        on a typo) since this is a manual debugging entry point, not the
        primary CLI surface -- that's main.py's `compare` subcommand."""
        names = list(DATASET_LOADERS.keys())
        print("Available datasets:")
        for i, name in enumerate(names, start=1):
            print(f"  {i}. {name}")
        choice = input(f"Choose a dataset [{'/'.join(names)}]: ").strip()
        if choice in DATASET_LOADERS:
            return choice
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        raise SystemExit(f"Unrecognized dataset {choice!r}.")

    # This standalone entry point requires a dataset explicitly, same as
    # main.py's compare/export subcommands -- comparing runs across
    # datasets is never allowed anywhere in this project's CLI surface.
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else None
    if dataset_name not in DATASET_LOADERS:
        if dataset_name is not None:
            print(f"Unrecognized dataset {dataset_name!r}.")
        dataset_name = _prompt_for_dataset_standalone()

    rows = build_comparison(dataset_name=dataset_name)

    timestamp = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")
    out_path = Path(f"results/comparison_{timestamp}_{dataset_name}.json")
    out_path.write_text(json.dumps(rows, indent=2))

    print(f"Compared {len(rows)} run(s) for dataset={dataset_name!r} -> {out_path}")
    