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
#
# summarize_run() extended 2026-07-31 (part 1): added final_hyperparameters
# and dataset/target/random_state, pulled from config -- see that
# session's notes for why.
#
# summarize_run() extended 2026-07-31 (part 2) -- REAL BUG FOUND AND
# FIXED: the "last evaluate_model in the log = the final model"
# heuristic was demonstrably wrong on a real file
# (result_log_2026_07_29_232959_breast_cancer.json) -- Gemini evaluated
# random_forest LAST purely as a comparison point, then its own
# convergence reasoning named an EARLIER logistic_regression evaluation
# (recall 0.9722) as the actual chosen model. Fixed by resolving the
# final model as whichever evaluated model scores highest on
# config["target"]'s metric (Option A, confirmed with Melanie), not by
# call order. This is not text-parsing of the free-text reasoning field
# (too fragile) -- it's a structural, metric-based resolution.
#
# Because this heuristic can still, in principle, disagree with what
# Gemini's reasoning text actually says (rare, but the breast_cancer
# file is proof it happens), summarize_run() now also flags
# final_model_ambiguous: True whenever the best-by-target pick differs
# from the old last-evaluated pick -- a cheap, structural mismatch
# signal, not an attempt to parse or adjudicate the reasoning itself.
# Deliberately NOT resolved by a second live Gemini call -- considered
# and explicitly deferred (2026-07-31): that idea overlaps with the
# still-undesigned human-in-the-loop hook (item #6) and
# Agent-decisions.md generator (item #12), and belongs in that future
# conversation, not bolted onto this module.

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The same four metrics evaluate_model (trainer.py) computes -- kept
# here as a local constant rather than importing VALID_TARGETS from
# main.py, since compare_runs.py must stay import-independent of
# main.py (see reporting.py's header comment for why).
KNOWN_METRICS = {"accuracy", "precision", "recall", "f1"}


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
    in their result dict at all. Runs from before 2026-07-29 have no
    top-level "config" block at all (dataset/target/random_state were
    not yet persisted there) -- which also means "final model"
    resolution for those older files falls back to "last evaluated",
    since there's no target metric to rank by. Every access below uses
    .get(...) with an explicit default rather than direct indexing.

    Returns a dict with these keys, always present (None/[]/empty when
    the source run doesn't have the underlying data):
      source_file, status, iterations, elapsed_seconds, dataset, target,
      random_state, model_sequence, final_model_type,
      final_hyperparameters, final_metrics, final_model_ambiguous,
      warnings_encountered, convergence_reasoning

    final_model resolution (CHANGED 2026-07-31, see module header for
    the real bug this fixes): every evaluate_model call this run is
    recorded, in order, as {model_ref, model_type, metrics}. If
    config["target"] is a known metric name, the "final" model is
    whichever evaluated model scores HIGHEST on that metric -- not
    whichever was evaluated last. If target is missing/unrecognized
    (older files), falls back to the original "last evaluated" rule
    unchanged.

    final_model_ambiguous (NEW 2026-07-31): True when target IS known
    and the best-by-target pick differs from the last-evaluated pick
    (a real disagreement worth a human's attention -- see
    reporting.py for how this surfaces in output). False when target
    is known and the two picks agree. None when target is unknown --
    there's no second opinion to compare against, so "ambiguous" isn't
    a meaningful question for those files.
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
            # Last one wins if there are several -- unchanged from
            # before; a run can call this with continue_iterating=True
            # more than once before finally stopping.
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
    result_log_*_<dataset_name>.json only -- confirmed with Melanie
    2026-07-29: comparing runs across different datasets is never
    allowed, since their metrics (accuracy/precision/recall/f1) aren't
    comparable against different data. main.py's `compare` and
    `export` subcommands always pass an explicit, validated
    dataset_name and never call this with None.

    dataset_name=None is still ACCEPTED by this function itself (so it
    stays a general-purpose, unopinionated utility) -- but as of
    2026-08-01, nothing in this project actually calls it that way
    anymore. RESOLVED (item #17): the standalone `python -m
    ml_agent.compare_runs` entry point (see this file's __main__ block)
    now requires a dataset too, matching main.py's `compare`/`export`
    subcommands -- comparing runs across datasets is never allowed
    anywhere in this project's CLI surface.

    Sorted by filename, which still sorts chronologically as-is, since
    the naming convention is
    result_log_<YYYY_MM_DD_HHMMSS>_<dataset_name>.json.
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

    # RESOLVED 2026-08-01 (item #17, previously genuinely open): this
    # standalone entry point now requires a dataset too, matching
    # main.py's `compare`/`export` -- confirmed with Melanie. Previously
    # called build_comparison() with no dataset_name at all, silently
    # mixing every dataset's runs into one file -- the exact thing the
    # rest of this project explicitly forbids elsewhere.
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