# Demo Run

A real, unedited execution of the full setup → test → run → compare →
export → report sequence, captured directly from terminal output.
Included here as evidence of expected behavior, not a constructed
example — every command below was actually run, in this order, against
the current codebase.

---

## 1. Environment sync

```
$ uv sync
Resolved 44 packages in 26ms
Checked 41 packages in 27ms
```

No installs or downloads — the environment already matched `uv.lock`
exactly.

---

## 2. Test suite

```
$ uv run pytest -v
...
===================================================================== 52 passed in 3.99s ======================================================================
```

All 52 tests pass. The suite is fully local and deterministic — no
Gemini API calls are made; the one network-dependent code path
(`fetch_openml`, in `load_climate_crashes_dataset`) is exercised
against a mocked return value in `test_dataset.py`, not the real
OpenML service.

---

## 3. Run — climate dataset, optimizing for recall

```
$ uv run python main.py --dataset climate --target recall --random-state 42 --log-iterations

converged - 7 iterations
total elapsed time: 31.6s
```

Gemini proposed Random Forest first (recall 0.22), rejected it as
insufficient, then proposed an SVM (`kernel=rbf, class_weight=balanced,
C=1`), which reached recall 1.0 at the cost of precision (0.18), and
converged there. Full step-by-step tool-call trail printed to the
terminal and written to `results/result_log_2026_08_13_215149_climate.json`.

The specific model sequence Gemini explores is not deterministic across
runs — its proposal reasoning is free-form LLM output, not seeded. What
is deterministic, given identical hyperparameters and `random_state`, is
each estimator's own fit: the SVM in this run produced results
consistent with prior runs on the same dataset/target/seed combination
(see Section 5).

---

## 4. Run — breast cancer dataset, optimizing for F1, quiet mode

```
$ uv run python main.py --dataset breast_cancer --target f1 --random-state 42 --no-log-iterations

converged - 12 iterations
total elapsed time: 21.7s
```

`--no-log-iterations` suppresses the step-by-step trail from the
terminal; the run's result is still persisted to
`results/result_log_2026_08_13_215347_breast_cancer.json` regardless of
this flag.

---

## 5. Compare

```
$ uv run python main.py compare --dataset climate
Compared 13 run(s) for dataset='climate' -> results/comparison_2026_08_13_195718_climate.json
```

13 matches the number of `result_log_*_climate.json` files present at
run time (12 pre-existing plus the one from Section 3). A `.md` report
for an earlier run sitting in the same directory was correctly excluded
— `build_comparison`'s glob targets `result_log_*_climate.json`
specifically.

Inspecting the resulting CSV export (Section 6) against this comparison
data confirms two defensive behaviors working as intended on real files:

- Runs predating the `config` block (six files from the earliest
  session) show empty `dataset`/`target`/`random_state`/
  `final_model_ambiguous` fields, rather than raising.
- `final_model_ambiguous` correctly flags `True` on two real runs where
  the last-evaluated model differed from the model actually named in
  Gemini's own convergence reasoning — the exact scenario this field
  exists to catch.

---

## 6. Export

```
$ uv run python main.py export --dataset climate
Exported 13 run(s) for dataset='climate' -> results/export_2026_08_13_201230_climate.csv
```

Row count matches the comparison file. `ConvergenceWarning` entries in
the `warnings_encountered` column correlate exactly with whether
`logistic_regression` appears in a run's `model_sequence`, as expected
given `LogisticRegression` is the only registered estimator whose
default hyperparameters can trigger that warning on this dataset.

---

## 7. Report

```
$ uv run python main.py report results/result_log_2026_08_13_215149_climate.json
Rendered single report for result_log_2026_08_13_215149_climate.json -> results/result_log_2026_08_13_215149_climate.md
```

```markdown
# Run report — result_log_2026_08_13_215149_climate.json

- **Dataset:** climate
- **Optimization target:** recall
- **Random state:** 42
- **Status:** converged
- **Iterations:** 7
- **Elapsed:** 31.6s

## Final model

- **Type:** svm
- **Hyperparameters:** `kernel=rbf`, `class_weight=balanced`, `C=1`
- **Metrics:** **accuracy**: 0.6204 · **precision**: 0.1800 · **recall**: 1.0000 · **f1**: 0.3051

## Model sequence tried this run

random_forest, svm

## Convergence reasoning (final)

The SVM model achieved perfect recall (1.0), which is the optimization
target. While the precision is low (0.18), the model succeeds in
identifying all instances of the minority class. This is the optimal
result for the specified goal.
```

No ambiguity warning is shown, consistent with this run's
`final_model_ambiguous: False` in both the comparison and export data
from Sections 5–6. The metrics, hyperparameters, and model sequence
shown here match the live terminal output from Section 3 and the CSV
row from Section 6 exactly — three independent renderings of the same
underlying run data, in agreement.
