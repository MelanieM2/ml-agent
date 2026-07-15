# Data Science Analysis: Agent Convergence Behavior Across Datasets

_Written 2026-07-13, following the first real end-to-end verification of
`gemini_client.py`'s agent loop. Based on 7 live runs (5 Climate Crashes, 2
Breast Cancer) against a live Gemini model. Findings below are grounded in
these actual runs — sample sizes are explicitly stated throughout, since
n=5 and n=2 are not large enough to support strong statistical claims; this
is an observational analysis of real agent behavior, not a controlled
experiment._

---

## 1. The dataset asymmetry this analysis depends on

The two datasets used were deliberately chosen for their **opposite class
semantics**:

| | Climate Crashes | Breast Cancer |
|---|---|---|
| `pos_label=1` means | simulation *failure* — rare (~8.5%), undesirable | *benign* — majority (~62.7%), desirable |
| Real-world cost asymmetry | missing a rare failure (false negative) is plausibly the expensive error | less clear-cut; both error types are more symmetric in typical framing |

This isn't incidental background — it's the throughline connecting every
finding below. The right metric to optimize (accuracy vs. precision vs.
recall) is genuinely different between these two datasets, and nothing in
the current agent setup tells Gemini which one applies.

---

## 2. Full run results

| # | Dataset | max_iter | Result | Iterations | Model(s) involved | Key reasoning |
|---|---|---|---|---|---|---|
| 1 | Climate | 2 | ceiling | 2 | — | plumbing check only, not a real search |
| 2 | Climate | 10 | **converged** | 8 | LogReg → Random Forest | RF chosen: accuracy 92.6%, precision 57.1%, F1 0.5 vs. 0.38 — despite recall *dropping* to 44.4% vs. LogReg |
| 3 | Climate | 10 | ceiling | 10 | (unknown which models tried) | non-deterministic vs. run 2, identical setup |
| 4 | Breast Cancer | 10 | **converged** | 4 | Logistic Regression | accuracy 95.6%, F1 0.966 — "further tuning unlikely to justify the extra complexity" |
| 5 | Breast Cancer | 10 | **converged** | 4 | Logistic Regression | accuracy 94.7%, F1 0.958 — consistent with run 4 in iteration count and model choice |
| 6 | Climate | 15 | **converged** | 12 | LogReg → Random Forest | RF chosen for best F1 (0.43) despite LogReg having better recall |
| 7 | Climate | ~10–15 (exact value not reconfirmed) | **converged** | 11 | LogReg → Random Forest → SVM | RF (F1 0.47) chosen over SVM despite SVM's perfect recall, explicitly citing "high cost of false positives" |

**Tally:** Climate: 5 real search runs, 3 converged (iterations 8, 12, 11),
2 hit the iteration ceiling at `max_iterations=10`. Breast Cancer: 2 runs,
both converged at iteration 4.

---

## 3. Finding: convergence speed and consistency track dataset difficulty

Breast Cancer converged in both attempts, at the identical iteration count
(4), with the identical model type (Logistic Regression), and closely
matching metrics (F1 0.966 vs. 0.958). Climate converged in only 3 of 5
attempts, at widely varying iteration counts (8, 12, 11) when it did
converge, and with reasoning that repeatedly wrestled with a genuine
precision/recall trade-off rather than reaching an obviously "good enough"
answer quickly.

This is a plausible and sensible pattern, not an anomaly to explain away:
breast-cancer classification is a well-studied, typically well-separated
problem in feature space, while Climate Crashes' extreme class imbalance
(8.5% positive) makes "what counts as good performance" a genuinely harder
question — reflected directly in how much longer and more variably the
agent had to reason about it.

## 4. Finding: the missing optimization target is a real, observed problem — not just a theoretical gap

Across every Climate run where a choice had to be made between models, the
agent's `record_convergence_decision` reasoning explicitly traded away
recall in favor of accuracy, precision, or F1:

- Run 2: chose Random Forest despite recall *dropping* to 44.4% (from
  presumably higher Logistic Regression recall), citing accuracy and F1
  instead.
- Run 6: chose Random Forest for best F1 (0.43) "despite Logistic
  Regression having better recall."
- Run 7: chose Random Forest over SVM's "perfect recall," explicitly citing
  the cost of false positives instead.

On a dataset where the positive class is a rare, safety-relevant failure
mode, recall (catching actual failures) is plausibly the metric that
matters most — a missed failure (false negative) is arguably worse than a
false alarm (false positive) in most simulation-crash-prediction framings.
The agent consistently reasoned toward a *different* trade-off, three times
in a row, using different specific justifications each time (accuracy, F1,
precision/false-positive-cost) — suggesting this isn't a one-off quirk but
a systematic consequence of `initial_context` never stating which metric
actually matters for this problem.

**This is the single most actionable finding from this session's runs**:
`record_model_proposal`'s own tool schema expects reasoning "given... the
optimization target" — but nothing in the current prompt-building logic
ever states one. The fix (adding an explicit target, e.g. "optimize for
recall on the failure class, given the cost of missed failures") is small,
but this session's real evidence — not just the theoretical schema gap —
is what makes the case for prioritizing it.

Also worth noting for balance: it's possible the agent's accuracy/F1-
leaning behavior reflects a *reasonable default prior* in the absence of
explicit guidance (optimizing overall correctness is not an unreasonable
default), rather than a "wrong" choice per se. The finding here is that the
metric choice is currently undetermined by the prompt, not that the agent's
specific choices were incorrect — that's a judgment call the optimization
target should make explicit, not something this analysis asserts on its
own authority.

## 5. Finding: non-determinism is real, but may be partly a budget artifact, not pure randomness

Runs 2 and 3 used identical code, dataset, and `max_iterations=10`, yet one
converged (iteration 8) and the other hit the ceiling (iteration 10)
without a `record_convergence_decision` ever firing. Read in isolation,
this looks like simple run-to-run variance in an LLM-orchestrated search —
expected, since Gemini's calls aren't seeded/deterministic by default.

Run 6, however, complicates that read: at `max_iterations=15`, the same
dataset converged at iteration **12** — past where a `max_iterations=10`
ceiling would have cut it off. This raises a genuine, unresolved question:
were runs 2/3's ceiling-hits actually *stuck* loops, or simply
*mid-convergence*, cut short by a ceiling that happened to be too low for
that particular exploration path? **One data point isn't enough to answer
this** — it's flagged here as an open question for future investigation
(e.g. running several more Climate trials at `max_iterations=15` or higher
to see how often convergence happens past iteration 10), not resolved by
this session's data.

## 6. Secondary finding: `LogisticRegression`'s convergence warning is systematic, not incidental

A `ConvergenceWarning` (`lbfgs failed to converge after 100 iteration(s)`)
appeared on **every single run that involved Logistic Regression** — both
datasets, converged and ceiling-hit runs alike, including runs 4 and 5
(Breast Cancer) where the model still ultimately performed well (F1 >
0.95) despite the warning. This confirms the issue is tied to
`LogisticRegression`'s default `max_iter=100` setting itself, independent
of which dataset it's applied to — not a Climate-specific data-scaling
problem as might otherwise be assumed. A real, if minor, fix candidate:
raise `max_iter`, scale input features, or expose `max_iter` as a tunable
hyperparameter in `list_available_models`'s schema.

## 7. Summary

| Question | Answer, with appropriate caveats |
|---|---|
| Does convergence behavior differ meaningfully between the two datasets? | Yes — Breast Cancer converged faster (4 iterations, both runs) and more consistently (same model both times) than Climate (variable iteration counts, mixed convergence, more model exploration). n=2 vs. n=5, so treat as suggestive, not conclusive. |
| Does the missing optimization target actually affect model selection? | Yes, observed directly across 3 separate Climate runs, each trading away recall for a different alternative metric with different stated reasoning. |
| Is the loop's non-determinism a bug? | No — expected LLM behavior — but it may be partly conflated with insufficient iteration budget; genuinely unresolved by current data. |
| Is the `ConvergenceWarning` dataset-specific? | No — confirmed present across both datasets, appears tied to `LogisticRegression`'s own default settings. |

This analysis is intentionally kept separate from `README.md` (which
carries only the compact summary table from the earlier synthetic
cross-dataset smoke test) and from `TECHNICAL_NOTES.md` (which covers an
unrelated, purely architectural question about conversation-history cost
scaling) — this file is specifically the methodology/data-science-layer
findings from real, live agent runs.
