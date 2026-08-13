# ml_agent/reporting.py — turns a single result_log_*.json (one run) or
# an already-built comparison_*.json (several runs) into either a CSV
# file (spreadsheet export -- the project's named gate before going
# public) or a Markdown report (human-friendly viewer, auto-detected by
# filename).
#
# Deliberately its own module, not folded into main.py or
# compare_runs.py: both output formats need to call into
# compare_runs.py's summarize_run() (imported below), but neither
# main.py nor compare_runs.py should import FROM this module or from
# each other. compare_runs.py's own standalone `python -m
# ml_agent.compare_runs` entry point stays exactly as decoupled from
# main.py as it was before this module existed.
#
# Auto-detect strategy: filename-prefix matching only -- result_log_*
# vs comparison_* -- with no fallback that inspects the JSON content
# itself. This assumes these files are never renamed by hand; a
# content-inspection fallback was considered and explicitly deferred --
# revisit this function if that assumption ever breaks.

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

from ml_agent.compare_runs import summarize_run

Kind = Literal["single", "comparison"]


def detect_kind(path: Path) -> Kind:
    """Filename-prefix auto-detect: result_log_* -> "single" (one raw
    run, not yet summarized), comparison_* -> "comparison" (already a
    list of summarize_run() rows, written by build_comparison()).

    Raises ValueError on anything else -- deliberately loud rather than
    guessing, since this project doesn't produce a third file shape.
    """
    name = path.name
    if name.startswith("result_log_"):
        return "single"
    if name.startswith("comparison_"):
        return "comparison"
    raise ValueError(
        f"Can't tell what kind of results file {path.name!r} is -- "
        "expected a result_log_*.json or comparison_*.json filename."
    )


def load_rows(path: Path) -> tuple[list[dict[str, Any]], Kind]:
    """Loads path and returns (rows, kind) -- always a list of flat,
    comparable rows, regardless of which kind of file this was.

    - "single": path is one raw result_log_*.json -- summarize_run()
      (compare_runs.py) is reused as-is to flatten it into exactly ONE
      row. Not reimplemented here -- see this module's header comment
      for why calling INTO compare_runs.py (never the reverse) is safe.
    - "comparison": path is already a list of summarize_run() rows
      (written by build_comparison()) -- loaded directly, no reshaping
      needed.
    """
    kind = detect_kind(path)
    data = json.loads(path.read_text())

    if kind == "single":
        rows = [summarize_run(data, source_file=path.name)]
    else:
        rows = data  # already a list of rows

    return rows, kind


# ---------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------

# Fixed column order -- not dict.keys() from the first row -- so every
# CSV this writes has the same columns in the same order regardless of
# which fields happen to be populated in a given row (e.g. an older run
# with no config block still gets a "target" column, just empty).
CSV_COLUMNS = [
    "source_file",
    "dataset",
    "target",
    "status",
    "iterations",
    "elapsed_seconds",
    "random_state",
    "final_model_type",
    "final_hyperparameters",
    "final_model_ambiguous",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "confusion_matrix",
    "model_sequence",
    "warnings_encountered",
    "convergence_reasoning",
]


def _flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    """CSV cells are plain strings, not nested dicts/lists --
    final_metrics, final_hyperparameters, confusion_matrix,
    model_sequence, and warnings_encountered all need reshaping.

    final_metrics is unpacked into its own top-level accuracy/precision/
    recall/f1/confusion_matrix columns (matches CSV_COLUMNS above)
    rather than staying nested -- easier to sort/filter by a single
    metric in a real spreadsheet, which is the whole point of exporting
    a CSV in the first place. Everything else that's still nested
    (hyperparameters, warnings) is serialized to a compact JSON string
    -- readable, and round-trippable with json.loads() if anyone needs
    it back as data.
    """
    metrics = row.get("final_metrics") or {}
    hp = row.get("final_hyperparameters")
    warnings_list = row.get("warnings_encountered")
    confusion = metrics.get("confusion_matrix")

    return {
        "source_file": row.get("source_file"),
        "dataset": row.get("dataset"),
        "target": row.get("target"),
        "status": row.get("status"),
        "iterations": row.get("iterations"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "random_state": row.get("random_state"),
        "final_model_type": row.get("final_model_type"),
        "final_hyperparameters": json.dumps(hp) if hp is not None else "",
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "confusion_matrix": json.dumps(confusion) if confusion is not None else "",
        "model_sequence": ", ".join(row.get("model_sequence") or []),
        "warnings_encountered": json.dumps(warnings_list) if warnings_list else "",
        "convergence_reasoning": row.get("convergence_reasoning") or "",
        "final_model_ambiguous": row.get("final_model_ambiguous"),
    }


def to_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Writes rows (as returned by load_rows()) to out_path as CSV.

    One physical CSV row per logical run -- a "comparison" file's N
    rows become N CSV rows; a "single" file's one row becomes a
    one-line CSV. Same shape either way, rather than a special
    transposed layout for single runs, so the CSV format is predictable
    regardless of which kind of file was exported.
    """
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(_flatten_for_csv(row))


# ---------------------------------------------------------------------
# Markdown report (the human-friendly viewer)
# ---------------------------------------------------------------------

def _format_hyperparameters(hp: dict[str, Any] | None) -> str:
    if not hp:
        return "_(not available)_"
    return ", ".join(f"`{k}={v}`" for k, v in hp.items())


def _format_metrics(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return "_(not available)_"
    parts = []
    for key in ("accuracy", "precision", "recall", "f1"):
        value = metrics.get(key)
        if value is not None:
            parts.append(f"**{key}**: {value:.4f}")
    return " · ".join(parts) if parts else "_(not available)_"


def _single_run_markdown(row: dict[str, Any]) -> str:
    """One result_log_*.json rendered as a Markdown report.

    Layout, deliberately in this order: config (dataset/target/seed --
    the run's starting conditions) first, then status/timing, then the
    final accepted model (type + hyperparameters + metrics -- the
    answer most readers want first), then the full model-sequence trail
    and any warnings for readers who want the journey, not just the
    destination.
    """
    elapsed = row.get("elapsed_seconds")
    elapsed_str = f"{elapsed:.1f}s" if elapsed is not None else "_(unknown)_"

    lines = [
        f"# Run report — {row.get('source_file') or '(unknown file)'}",
        "",
        f"- **Dataset:** {row.get('dataset') or '_(unknown)_'}",
        f"- **Optimization target:** {row.get('target') or '_(unknown)_'}",
        f"- **Random state:** {row.get('random_state')}",
        f"- **Status:** {row.get('status')}",
        f"- **Iterations:** {row.get('iterations')}",
        f"- **Elapsed:** {elapsed_str}",
        "",
        "## Final model",
        "",
        f"- **Type:** {row.get('final_model_type') or '_(none accepted)_'}",
        f"- **Hyperparameters:** {_format_hyperparameters(row.get('final_hyperparameters'))}",
        f"- **Metrics:** {_format_metrics(row.get('final_metrics'))}",
    ]

    if row.get("final_model_ambiguous"):
        lines.append(
            "- ⚠️ **Note:** this run evaluated a different model AFTER "
            "the one selected here -- the model shown above scored "
            "highest on the run's target metric, but wasn't the last "
            "one evaluated. Worth reading the convergence reasoning "
            "below carefully."
        )

    lines += [
        "",
        "## Model sequence tried this run",
        "",
        (", ".join(row.get("model_sequence") or []) or "_(none recorded)_"),
        "",
        "## Convergence reasoning (final)",
        "",
        row.get("convergence_reasoning") or "_(none recorded)_",
    ]

    warnings_list = row.get("warnings_encountered") or []
    if warnings_list:
        lines += ["", "## Warnings encountered", ""]
        for w in warnings_list:
            first_line = (w.get("message") or "").splitlines()[0] if w.get("message") else ""
            lines.append(
                f"- iteration {w.get('iteration')}, `{w.get('model_type')}`: "
                f"**{w.get('category')}** — {first_line}"
            )

    return "\n".join(lines) + "\n"


def _comparison_markdown(rows: list[dict[str, Any]], source_file: str) -> str:
    """Several runs (already-flattened comparison rows) rendered as one
    Markdown table -- GitHub renders tables natively, so this needs no
    extra tooling for a reader browsing the repo.
    """
    header = (
        "| Run | Dataset | Target | Model | Hyperparameters | "
        "Accuracy | Precision | Recall | F1 | Iterations | Status | ⚠️ |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [f"# Comparison report — {source_file}", "", header, sep]

    for row in rows:
        metrics = row.get("final_metrics") or {}

        def fmt(key: str) -> str:
            value = metrics.get(key)
            return f"{value:.4f}" if value is not None else "?"

        flag = "⚠️" if row.get("final_model_ambiguous") else ""

        lines.append(
            "| {src} | {ds} | {tgt} | {model} | {hp} | {acc} | {prec} | {rec} | {f1} | {iters} | {status} | {flag} |".format(
                src=row.get("source_file") or "?",
                ds=row.get("dataset") or "?",
                tgt=row.get("target") or "?",
                model=row.get("final_model_type") or "?",
                hp=_format_hyperparameters(row.get("final_hyperparameters")),
                acc=fmt("accuracy"),
                prec=fmt("precision"),
                rec=fmt("recall"),
                f1=fmt("f1"),
                iters=row.get("iterations"),
                status=row.get("status"),
                flag=flag,
            )
        )

    return "\n".join(lines) + "\n"


def to_markdown(rows: list[dict[str, Any]], kind: Kind, source_file: str) -> str:
    """Dispatches to the single-run or comparison renderer based on
    kind (as returned by load_rows()) -- the one place that decision
    is made, so main.py's `report` subcommand never needs to know the
    difference between the two file shapes.
    """
    if kind == "single":
        return _single_run_markdown(rows[0])
    return _comparison_markdown(rows, source_file)
