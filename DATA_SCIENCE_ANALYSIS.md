# Data Science Analysis: Agent Convergence Behavior Across Datasets

_Companion to [`TECHNICAL_NOTES.md`](./TECHNICAL_NOTES.md) — this file covers methodology and findings; that one covers implementation decisions. Most working sessions touched both, on the same date, about the same piece of work — see the session map below._

---

## Session map: how this file lines up with `TECHNICAL_NOTES.md`

```
TECHNICAL_NOTES.md                                 DATA_SCIENCE_ANALYSIS.md (this file)
──────────────────────────────────────────────────────────────────────────
Part 1   2026-07-13  eval & deferred, not built    ·   (no DS counterpart)
Part 2   2026-07-17  orchestration-loop wiring     ──▶ §8   optimization-target fix, tested live
Part 3   2026-07-19  per-iteration logging         ──▶ §9   first look inside a run
Part 4   2026-07-24  warning capture + compare     ──▶ §10  cross-run comparison goes live
Part 5   2026-07-27  max_iter exposed              ──▶ §11  max_iter tested against 3 live runs
Part 6   2026-07-29  reproducibility fix            ──▶ §12  max_iter-vs-C confound resolved
Part 7   2026-08-01  reporting.py, final-model fix  ──▶ §13  final-model resolution corrected
Part 8   2026-08-03  test_trainer.py, 17 tests     ─ ─ ▶ §13  (cross-reference only, no new run)
```

---

## Status index

| § | Date | Covers | Status | See also |
|---|---|---|---|---|
| [1](#1-the-dataset-asymmetry-this-analysis-depends-on) | — | Dataset asymmetry (Climate Crashes vs. Breast Cancer) this analysis depends on | Foundational framing | — |
| [2](#2-full-run-results) | — | Full run results | — | — |
| [3](#3-finding-convergence-speed-and-consistency-track-dataset-difficulty) | — | Convergence speed/consistency tracks dataset difficulty | Finding | — |
| [4](#4-finding-the-missing-optimization-target-is-a-real-observed-problem--not-just-a-theoretical-gap) | — | Missing optimization target is a real, observed problem | Finding — fixed & verified in §8 | [TN Part 2](./TECHNICAL_NOTES.md#part-2-orchestration-loop-wiring--implementation-details-2026-07-17) |
| [5](#5-finding-non-determinism-is-real-but-may-be-partly-a-budget-artifact-not-pure-randomness) | — | Non-determinism, partly a budget artifact | Finding | — |
| [6](#6-secondary-finding-logisticregressions-convergence-warning-is-systematic-not-incidental) | — | `LogisticRegression` convergence warning is systematic | Finding — resolved in §11 | [TN Part 5](./TECHNICAL_NOTES.md#part-5-max_iter-exposed-as-a-hyperparameter-confirming-the-agent-reasoning-pathway-was-already-wired-2026-07-27) |
| [7](#7-summary-as-of-2026-07-13) | 2026-07-13 | Summary as of this date | — | — |
| [8](#8-update-2026-07-17-the-optimization-target-fix-tested-for-the-first-time-against-a-live-agent) | 2026-07-17 | Optimization-target fix, tested live | Verified | [TN Part 2](./TECHNICAL_NOTES.md#part-2-orchestration-loop-wiring--implementation-details-2026-07-17) |
| [9](#9-update-2026-07-19-per-iteration-logging--first-look-inside-a-runs-actual-search-path) | 2026-07-19 | Per-iteration logging — first look inside a run | Verified | [TN Part 3](./TECHNICAL_NOTES.md#part-3-per-iteration-logging-in-run_agent_loop-2026-07-19) |
| [10](#10-update-2026-07-24-cross-run-comparison-goes-live--the-8493-question-finally-has-real-data-behind-it) | 2026-07-24 | Cross-run comparison goes live | Verified | [TN Part 4](./TECHNICAL_NOTES.md#part-4-fit-time-warning-capture-corrected-run-persistence-and-cross-run-comparison-2026-07-24) |
| [11](#11-update-2026-07-27-max_iter-exposed-as-a-tunable-hyperparameter--the-convergencewarning-shortlist-decision-tested-against-3-live-runs) | 2026-07-27 | `max_iter` exposed, tested against 3 live runs | Verified | [TN Part 5](./TECHNICAL_NOTES.md#part-5-max_iter-exposed-as-a-hyperparameter-confirming-the-agent-reasoning-pathway-was-already-wired-2026-07-27) |
| [12](#12-update-2026-07-29-the-max_iter-vs-c-confound-resolved-117s-anomaly-explained-and-new-breast-cancer-results) | 2026-07-29 | `max_iter`-vs-`C` confound resolved; reproducibility bug found & fixed | Verified | [TN Part 6](./TECHNICAL_NOTES.md#part-6-results-file-renaming-compare-subcommand-and-the-reproducibility-fix-2026-07-29) |
| [13](#13-update-2026-08-01-summarize_runs-final-model-resolution-corrected--a-real-breast_cancer-runs-reported-outcome-was-wrong) | 2026-08-01 | `summarize_run`'s final-model resolution corrected | Verified | [TN Part 7](./TECHNICAL_NOTES.md#part-7-results-reporting-reportingpy-a-real-final-model-bug-fix-and-a-pytest-collection-bug-2026-08-01), cross-ref [TN Part 8](./TECHNICAL_NOTES.md#part-8-teststest_trainerpy--reproducibility-test-coverage-and-an-empirical-debugging-chain-2026-08-03-session) |

---

## Table of contents

- [1. The dataset asymmetry this analysis depends on](#1-the-dataset-asymmetry-this-analysis-depends-on)

- [2. Full run results](#2-full-run-results)

- [3. Finding: convergence speed and consistency track dataset difficulty](#3-finding-convergence-speed-and-consistency-track-dataset-difficulty)

- [4. Finding: the missing optimization target is a real, observed problem — not just a theoretical gap](#4-finding-the-missing-optimization-target-is-a-real-observed-problem--not-just-a-theoretical-gap)

- [5. Finding: non-determinism is real, but may be partly a budget artifact, not pure randomness](#5-finding-non-determinism-is-real-but-may-be-partly-a-budget-artifact-not-pure-randomness)

- [6. Secondary finding: `LogisticRegression`'s convergence warning is systematic, not incidental](#6-secondary-finding-logisticregressions-convergence-warning-is-systematic-not-incidental)

- [7. Summary (as of 2026-07-13)](#7-summary-as-of-2026-07-13)

<!-- -->

<details>
<summary><a href="#8-update-2026-07-17-the-optimization-target-fix-tested-for-the-first-time-against-a-live-agent">8. Update, 2026-07-17: the optimization-target fix, tested for the first time against a live agent</a></summary>

- [8.1 Results](#81-results)
- [8.2 Finding: the fix worked, directly and repeatably — a genuine before/after contrast with §4](#82-finding-the-fix-worked-directly-and-repeatably--a-genuine-beforeafter-contrast-with-4)
- [8.3 Caveat: recall = 1.0 deserves a skeptical read, not just acceptance as "the fix working perfectly"](#83-caveat-recall--10-deserves-a-skeptical-read-not-just-acceptance-as-the-fix-working-perfectly)
- [8.4 Open question: run-to-run variation in iteration count and `ConvergenceWarning`, unresolved without logging](#84-open-question-run-to-run-variation-in-iteration-count-and-convergencewarning-unresolved-without-logging)
- [8.5 Minor, non-technical observation](#85-minor-non-technical-observation)
- [8.6 Updated summary table](#86-updated-summary-table)

</details>

<!-- -->

<!-- -->

<details>
<summary><a href="#9-update-2026-07-19-per-iteration-logging--first-look-inside-a-runs-actual-search-path">9. Update, 2026-07-19: per-iteration logging — first look inside a run's actual search path</a></summary>

- [9.1 One new logged run, Climate Crashes, `optimization_target="recall"`](#91-one-new-logged-run-climate-crashes-optimization_targetrecall)
- [9.2 Finding: the log confirms the model-search pattern implied, but not directly observed, in §8](#92-finding-the-log-confirms-the-model-search-pattern-implied-but-not-directly-observed-in-8)
- [9.3 The original §8.4 question: still not answered, and here's precisely why](#93-the-original-84-question-still-not-answered-and-heres-precisely-why)
- [9.4 Reconfirming §8.3's precision caveat](#94-reconfirming-83s-precision-caveat)
- [9.5 Updated summary table](#95-updated-summary-table)

</details>

<!-- -->

<!-- -->

<details>
<summary><a href="#10-update-2026-07-24-cross-run-comparison-goes-live--the-8493-question-finally-has-real-data-behind-it">10. Update, 2026-07-24: cross-run comparison goes live — the §8.4/§9.3 question finally has real data behind it</a></summary>

- [10.1 Six real runs, Climate Crashes, `optimization_target="recall"`](#101-six-real-runs-climate-crashes-optimization_targetrecall)
- [10.2 Finding: the warning/iteration-count variation is explained by *which models get proposed at all*, not by the warning alone](#102-finding-the-warningiteration-count-variation-is-explained-by-which-models-get-proposed-at-all-not-by-the-warning-alone)
- [10.3 Correction: what `warnings.simplefilter("always")` actually protects against](#103-correction-what-warningssimplefilteralways-actually-protects-against)
- [10.4 Reconfirming the precision caveat, again](#104-reconfirming-the-precision-caveat-again)
- [10.5 Known limitation: a failed run is invisible to this comparison, not just excluded from it](#105-known-limitation-a-failed-run-is-invisible-to-this-comparison-not-just-excluded-from-it)
- [10.6 Updated summary table](#106-updated-summary-table)

</details>

<!-- -->

<!-- -->

<details>
<summary><a href="#11-update-2026-07-27-max_iter-exposed-as-a-tunable-hyperparameter--the-convergencewarning-shortlist-decision-tested-against-3-live-runs">11. Update, 2026-07-27: `max_iter` exposed as a tunable hyperparameter — the ConvergenceWarning shortlist decision, tested against 3 live runs</a></summary>

- [11.1 The shortlist, and the decision made](#111-the-shortlist-and-the-decision-made)
- [11.2 A caveat stated before, not after, seeing results](#112-a-caveat-stated-before-not-after-seeing-results)
- [11.3 Three live runs, Climate Crashes, `optimization_target="recall"`, `max_iter` now exposed](#113-three-live-runs-climate-crashes-optimization_targetrecall-max_iter-now-exposed)
- [11.4 Finding: the mechanism works, exactly as designed — documented fact, not inference](#114-finding-the-mechanism-works-exactly-as-designed--documented-fact-not-inference)
- [11.5 Finding: Run A isolates `max_iter`'s own effect cleanly; Runs B/C confound it with `C`](#115-finding-run-a-isolates-max_iters-own-effect-cleanly-runs-bc-confound-it-with-c)
- [11.6 Finding: a new answer that dominates the project's prior typical outcome](#116-finding-a-new-answer-that-dominates-the-projects-prior-typical-outcome)
- [11.7 Secondary observation, flagged as curious, not explained](#117-secondary-observation-flagged-as-curious-not-explained)
- [11.8 Updated summary table](#118-updated-summary-table)

</details>

<!-- -->

<!-- -->

<details>
<summary><a href="#12-update-2026-07-29-the-max_iter-vs-c-confound-resolved-117s-anomaly-explained-and-new-breast-cancer-results">12. Update, 2026-07-29: the `max_iter`-vs-`C` confound resolved, §11.7's anomaly explained, and new Breast Cancer results</a></summary>

- [12.1 Correction: the metric that actually moved was recall, not precision](#121-correction-the-metric-that-actually-moved-was-recall-not-precision)
- [12.2 The isolation run](#122-the-isolation-run)
- [12.3 §11.7's anomaly, resolved: not a curiosity, a real reproducibility bug](#123-117s-anomaly-resolved-not-a-curiosity-a-real-reproducibility-bug)
- [12.4 New results: Breast Cancer across all three model types, post-fix](#124-new-results-breast-cancer-across-all-three-model-types-post-fix)
- [12.5 Updated summary table](#125-updated-summary-table)

</details>

<!-- -->

<!-- -->

<details>
<summary><a href="#13-update-2026-08-01-summarize_runs-final-model-resolution-corrected--a-real-breast_cancer-runs-reported-outcome-was-wrong">13. Update, 2026-08-01: `summarize_run`'s final-model resolution corrected — a real breast_cancer run's reported outcome was wrong</a></summary>

- [13.1 What was assumed, and why it was wrong](#131-what-was-assumed-and-why-it-was-wrong)
- [13.2 The real run](#132-the-real-run)
- [13.3 The fix, and what it changes about how to read this document](#133-the-fix-and-what-it-changes-about-how-to-read-this-document)
- [13.4 Deferred: what happens when the flag actually fires](#134-deferred-what-happens-when-the-flag-actually-fires)
- [13.5 Summary table](#135-summary-table)

</details>

<!-- -->

---

_Written 2026-07-13, following the first real end-to-end verification of
`gemini_client.py`'s agent loop. Based on 7 live runs (5 Climate Crashes, 2
Breast Cancer) against a live Gemini model. Findings below are grounded in
these actual runs — sample sizes are explicitly stated throughout, since
n=5 and n=2 are not large enough to support strong statistical claims; this
is an observational analysis of real agent behavior, not a controlled
experiment._

_Updated 2026-07-17 — see §8 for new findings from the first 3 live runs made possible by that session's orchestration-loop wiring_

_Updated 2026-07-19 — see §9: per-iteration logging — first look inside a run's actual search path_

_Updated 2026-07-24 — see §10: cross-run comparison goes live — the §8.4/§9.3 question finally has real, comparable data behind it_

_Updated 2026-07-27 — see §11: `max_iter` exposed as a tunable hyperparameter — the ConvergenceWarning shortlist decision, tested against 3 live runs_

_Updated 2026-07-29 — see §12: the `max_iter`-vs-`C` confound isolated and resolved, §11.7's flagged Random Forest anomaly explained (a real reproducibility bug, not a curiosity), and new Breast Cancer results across all three model types_

_Updated 2026-08-01 — see §13: summarize run's final- model resolution corrected — a real breast cancer run's reported outcome was wrong_

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

## 7. Summary (as of 2026-07-13)

| Question | Answer, with appropriate caveats |
|---|---|
| Does convergence behavior differ meaningfully between the two datasets? | Yes — Breast Cancer converged faster (4 iterations, both runs) and more consistently (same model both times) than Climate (variable iteration counts, mixed convergence, more model exploration). n=2 vs. n=5, so treat as suggestive, not conclusive. |
| Does the missing optimization target actually affect model selection? | Yes, observed directly across 3 separate Climate runs, each trading away recall for a different alternative metric with different stated reasoning. |
| Is the loop's non-determinism a bug? | No — expected LLM behavior — but it may be partly conflated with insufficient iteration budget; genuinely unresolved by current data. |
| Is the `ConvergenceWarning` dataset-specific? | No — confirmed present across both datasets, appears tied to `LogisticRegression`'s own default settings. |

---

## 8. Update, 2026-07-17: the optimization-target fix, tested for the first time against a live agent

The 2026-07-17 session wired `agent.py`'s orchestration loop end-to-end
for the first time (`build_dispatch_table` → `run_session` →
`gemini_client.run_agent_loop`) and, as part of that work, directly acted
on §4's finding above: `initial_context` now states an explicit
`optimization_target` (see `README.md`'s "Core architectural principle"
section, and `TECHNICAL_NOTES.md` §"2026-07-17" for the implementation).
This section reports the first real evidence of what that fix actually
does to agent behavior — 3 live runs, Climate Crashes only,
`optimization_target="recall"`, via the new `run_smoke_test.py`.

### 8.1 Results

| Run | Model chosen | Recall | Precision | Iterations | `max_iterations` | `ConvergenceWarning`? |
|---|---|---|---|---|---|---|
| 1 | SVM, `class_weight="balanced"` | 1.0 | low (unspecified exact value) | 10 | 15 | Yes (Logistic Regression) |
| 2 | SVM, `class_weight="balanced"` | 1.0 | low (unspecified exact value) | 10 | 15 | Yes (Logistic Regression) |
| 3 | SVM, `C=0.1`, `class_weight="balanced"` | 1.0 | low (unspecified exact value) | 8 | 15 | No |

n=3, all Climate Crashes, all same `optimization_target`. Not yet run
against Breast Cancer or a different target (e.g. `"precision"`) — both
flagged as natural next data points, not done this session.

### 8.2 Finding: the fix worked, directly and repeatably — a genuine before/after contrast with §4

All three runs independently converged on an SVM with
`class_weight="balanced"` achieving **recall = 1.0**, with each run's own
`record_convergence_decision` reasoning explicitly referencing the stated
target (e.g. *"achieved a recall of 1.0, which is the maximum possible for
this metric... meets the requirement of optimizing for recall"*). This is
a direct, sharp contrast with §4's pre-fix findings, where three separate
Climate runs each independently traded recall *away* for a different
metric, with no stated target to anchor the decision. Three independent
post-fix runs landing on the same recall-maximizing conclusion, using
consistent language tied to the stated target, is real evidence the fix
changes agent behavior in the intended direction — not just that the
prompt text changed.

### 8.3 Caveat: recall = 1.0 deserves a skeptical read, not just acceptance as "the fix working perfectly"

All three runs also report **low precision**, and the agent's own
reasoning acknowledges this without really interrogating it — e.g. run 3's
reasoning states *"this is expected given the extreme imbalance and the
focus on recall"* and treats that as sufficient justification. With
`class_weight="balanced"` and a very small rare-class test set (9 examples,
per `validate_split`'s threshold-of-5 reasoning — see `README.md`
"Statistical rationale"), a model can sometimes reach perfect recall simply
by predicting the positive class liberally, catching every real case at
the cost of many false alarms. Whether "recall at any precision cost" is
actually the *right* trade-off for this problem is a human judgment call,
not something the current setup asks the agent to weigh — the fix
successfully made the agent optimize for exactly what was asked, but
"exactly what was asked" may itself be too blunt an instruction. A natural
follow-up (not built this session): a richer target that also expresses a
precision floor or an explicit cost ratio between false negatives and
false positives, rather than a single bare metric name.

### 8.4 Open question: run-to-run variation in iteration count and `ConvergenceWarning`, unresolved without logging

Run 3 converged in 8 iterations with no `ConvergenceWarning` at all, versus
10 iterations with the warning present in runs 1 and 2. This could mean
Logistic Regression wasn't proposed in run 3, or was proposed with
different hyperparameters that avoided the 100-iteration cap — but this
can't be confirmed from the final printed result alone, since
`run_agent_loop` currently returns only the final decision, not the full
per-iteration proposal/evaluation history.

**This is now a planned, explicitly requested piece of future work**:
optional logging of each iteration's tool call (name + arguments + result)
inside `run_agent_loop`, so questions like this one can be answered
directly from a run's log rather than left as an open question. Not built
this session — flagged for a future session, see `context_ml-agent_2026-07-17.md`.

### 8.5 Minor, non-technical observation

Run 3's `next_step_hint` field contained two missing-space typos
(`"optimalrecall"`, `"mosteffective"`) — this is Gemini's own free-text
output, passed through verbatim by `record_convergence_decision`, not a
defect in any project code. Noted for completeness, not actionable.

### 8.6 Updated summary table

| Question | Answer, with appropriate caveats |
|---|---|
| Does stating an explicit optimization target change agent behavior? | Yes — 3/3 post-fix runs converged on the recall-maximizing model with reasoning explicitly tied to the stated target, versus 3/3 pre-fix runs (§4) that traded recall away with no target to anchor the decision. n=3 vs. n=3, same caveats about sample size apply as elsewhere in this document. |
| Is recall = 1.0 unambiguously the "right" answer? | Not necessarily — precision was low in all 3 runs and largely unexamined by the agent's own reasoning. The fix delivers what was asked, which may itself need refining (e.g. a stated precision floor or explicit cost ratio) rather than a bare metric name. |
| Can today's data explain the iteration-count/warning variation between runs? | No — this requires per-iteration logging, not yet built. Flagged as planned future work. |

---

<!--
DATA_SCIENCE_ANALYSIS.md update — 2026-07-19 session
-->

## 9. Update, 2026-07-19: per-iteration logging — first look inside a run's actual search path

§8.4 flagged an unanswered question: run 3 (of the 3-run set in §8)
converged in 8 iterations with no `ConvergenceWarning`, versus 10
iterations with the warning present in runs 1 and 2 — but the final
printed result alone couldn't say why, since `run_agent_loop` returned
only the final decision. This session built optional per-iteration
logging (`log_iterations=True`; implementation in
`TECHNICAL_NOTES.md` Part 3) to close that gap. This section reports
what a first real logged run actually shows.

### 9.1 One new logged run, Climate Crashes, `optimization_target="recall"`

| Run | Model chosen | Recall | Precision | Iterations | `max_iterations` | `ConvergenceWarning`? |
|---|---|---|---|---|---|---|
| 8 | SVM, `kernel="rbf"`, `C=1`, `class_weight="balanced"` | 1.0 | 0.18 | 12 | 15 | Yes (during Logistic Regression's `train_model` call) |

n=1 for this specific logged run — this section is about what the *log
itself* reveals, not a new statistical claim; it should be read as a
qualitative complement to §8's n=3 findings, not an addition to that
sample.

### 9.2 Finding: the log confirms the model-search pattern implied, but not directly observed, in §8

With full visibility into the run for the first time, the actual
sequence was:

1. `list_available_models` (iteration 0)
2. **Logistic Regression**, `class_weight="balanced"` proposed and
   evaluated: recall 0.78, precision 0.25 — reasoned as "a strong
   baseline," but flagged by the agent's own convergence reasoning as
   having "significant room for improvement, especially in precision"
   (iterations 1–4)
3. **Random Forest**, `n_estimators=200`, `max_depth=10`,
   `class_weight="balanced"` proposed next: recall *dropped* to 0.22,
   precision rose to 0.4 — explicitly reasoned by the agent as "dropped
   in recall significantly, making it unsuitable for the stated
   optimization target" (iterations 5–8)
4. **SVM**, `kernel="rbf"`, `C=1`, `class_weight="balanced"` proposed
   third: recall 1.0, precision 0.18 — accepted, converged (iterations
   9–12)

This is the first time this project has had direct evidence of an
agent *rejecting* an intermediate model specifically because it
regressed on the stated optimization target (step 3), rather than
inferring that kind of behavior indirectly from a final chosen model's
metrics, as §8's pre-logging runs required. It's a small but concrete
piece of evidence that the fix from §8 doesn't just produce a
recall-maximizing final answer — the agent's *intermediate* reasoning,
not only its final one, is legibly tracking the stated target
throughout the search, not just at the end.

### 9.3 The original §8.4 question: still not answered, and here's precisely why

This one logged run does not, by itself, explain why run 3 (§8)
converged faster with no warning than runs 1–2. That comparison needs
multiple runs' *logs* set side by side — this session built the
capability to produce a log per run, but not yet a way to persist and
compare several runs' logs against each other (§8.4's original ask was
specifically about *cross*-run comparison, not single-run visibility).
The `ConvergenceWarning` was pinpointed for *this* run, though, as a
useful side effect: it's now confirmed to fire specifically during
Logistic Regression's `train_model` call (iteration 2 in this run's
log), consistent with §6's earlier finding that the warning is tied to
`LogisticRegression`'s own default `max_iter=100`, not to any
particular dataset.

**Status: cross-run comparison remains open, flagged as future work**
(persisted, comparable log output across multiple runs — format not yet
decided) — see `TECHNICAL_NOTES.md` Part 3 and
`context_ml-agent_2026-07-19.md`.

### 9.4 Reconfirming §8.3's precision caveat

This run's precision (0.18) is consistent with §8's pattern — recall
optimization, unqualified by any stated precision floor or cost ratio,
continues to produce models that catch every real case at a real cost
in false alarms. §8.3's proposed follow-up (a richer target expressing
a precision floor or an explicit false-negative/false-positive cost
ratio, rather than a bare metric name) remains unbuilt and, based on
this additional data point, still seems like the right next refinement
if this behavior needs adjusting.

### 9.5 Updated summary table

| Question | Answer, with appropriate caveats |
|---|---|
| Does the logged model-search trail confirm §8's optimization-target finding? | Yes — this run shows the agent explicitly rejecting an intermediate model (Random Forest) for regressing on the stated target, not just accepting a final model consistent with it. n=1 for this specific observation. |
| Does this resolve §8.4's cross-run question? | No — that needs multiple runs' logs compared, which the logging feature enables but does not yet automate; still open. |
| Does the precision caveat from §8.3 still hold? | Yes, reconfirmed on this additional run (precision 0.18). |

---

<!--
DATA_SCIENCE_ANALYSIS.md update — 2026-07-24 session
-->

## 10. Update, 2026-07-24: cross-run comparison goes live — the §8.4/§9.3 question finally has real data behind it

§9.3 left the original §8.4 question explicitly open: run-to-run
variation in iteration count and `ConvergenceWarning` presence needed
*multiple* runs' logs compared side by side, which the logging feature
enabled but didn't yet automate. This session built that missing piece
(`ml_agent/compare_runs.py`; implementation in `TECHNICAL_NOTES.md`
Part 4) and generated six fresh live runs to test it against. This
section reports what comparing them actually shows — a real answer,
though a more qualified one than "warning vs. no warning" alone.

### 10.1 Six real runs, Climate Crashes, `optimization_target="recall"`

| Run (local time) | Model sequence | Final recall / precision | Iterations | `ConvergenceWarning`? | `elapsed_seconds` |
|---|---|---|---|---|---|
| 23:00:36 | Random Forest → SVM (linear) | 0.778 / 0.25 | 8 | No | not yet captured (built later this session) |
| 23:17:59 | LogReg → Random Forest → SVM (rbf) | 1.0 / 0.18 | 12 | Yes | not yet captured |
| 23:36:53 | LogReg → Random Forest → SVM (rbf) | 1.0 / 0.18 | 10 | Yes | not yet captured |
| 23:39:38 | LogReg → Random Forest → SVM (rbf) | 1.0 / 0.18 | 10 | Yes | not yet captured |
| 23:56:33 | LogReg → Random Forest → SVM (rbf) | 1.0 / 0.18 | 10 | Yes | 8.4 |
| 23:57:39 | LogReg → Random Forest → SVM (rbf) | 1.0 / 0.18 | 10 | Yes | 8.1 |
| (failed) | — | — | — | 429 rate-limit; crashed before any file was written | — |

n=6 completed runs, one incomplete attempt excluded (see §10.5). All six
generated in a single evening, same dataset, same target, same code —
this is closer to a controlled repeat than most of this document's
earlier findings, but still not a designed experiment (no seed control
over Gemini's own sampling).

### 10.2 Finding: the warning/iteration-count variation is explained by *which models get proposed at all*, not by the warning alone

Five of six runs proposed **Logistic Regression first**, hit the same
`ConvergenceWarning`, then moved through Random Forest to a final SVM
with `kernel="rbf"` and recall = 1.0. The sixth run never proposed
Logistic Regression at all — it went straight to Random Forest, then an
SVM with `kernel="linear"` — converged two iterations sooner, and landed
on a *lower* final recall (0.778 vs. 1.0).

This refines §6's finding rather than contradicting it: the warning is
still tied to `LogisticRegression`'s own default `max_iter=100`,
independent of dataset (§6 remains correct as far as it goes) — but
*whether a given run encounters the warning at all* now looks like it
depends on whether Gemini proposes Logistic Regression in the first
place, which is itself the same kind of run-to-run non-determinism §5
already flagged as open and unresolved. **This is a plausible inference
from six data points, not a proven cause** — it would take a
substantially larger sample, or some way of inspecting *why* Gemini
chose its first proposal, to state this with real confidence.

### 10.3 Correction: what `warnings.simplefilter("always")` actually protects against

An earlier explanation this session (given conversationally, before this
write-up) described the risk `simplefilter("always")` guards against as
spanning *separate runs* of `run_smoke_test.py` — e.g., "a second run
hitting the identical warning could go undetected." **That framing was
imprecise, and is corrected here rather than left standing.**

Each invocation of `uv run python run_smoke_test.py` starts a fresh
Python process. Python's warning-deduplication registry
(`__warningregistry__`) lives in the memory of the module that issues
the warning, for the lifetime of that process — so it already resets
naturally between separate terminal invocations, with or without
`"always"`. None of tonight's six runs could have suppressed each
other's warnings even under Python's default filter, since each was
launched as its own process.

**What `"always"` actually protects against is narrower, and still
real:** any case where the *same warning could fire more than once
within a single process* — for instance, a future `main.py` that runs
several sessions in one long-lived process without restarting between
them, a `pytest` run exercising Logistic Regression training in more
than one test, or a single agent run where Gemini happens to propose
Logistic Regression more than once before converging. Without
`"always"`, only the *first* such occurrence in a process would ever be
recorded — every later one would silently vanish from `train_model`'s
returned `warnings` list, with no error, no missing key, nothing to
indicate data had been dropped. For a project whose entire point in this
area is building a reliable record of what actually happened during a
run, a filter that could silently under-report is a correctness risk
worth the two extra lines, even though it happened not to matter for any
of tonight's six separately-invoked runs specifically.

### 10.4 Reconfirming the precision caveat, again

All five runs that reached recall = 1.0 did so with the same precision,
0.18, and the same confusion matrix (`[[58, 41], [0, 9]]`) — identical
down to the exact figures, not just similar. This is a striking degree
of consistency for an LLM-orchestrated, non-deterministic search, and is
worth flagging plainly: it suggests that once a run reaches "SVM,
`kernel=rbf`, `class_weight=balanced`, `C=1`" on this exact dataset and
split (`random_state=42` is the default, unchanged across all runs
tonight), the *model's* behavior is fully deterministic even when the
*path to reach it* wasn't. §8.3's caveat still stands unchanged: recall
= 1.0 continues to come at a real, largely unexamined cost in false
alarms, and the richer-target follow-up proposed there (a stated
precision floor, or an explicit false-negative/false-positive cost
ratio) remains unbuilt and still looks like the right next refinement.

### 10.5 Known limitation: a failed run is invisible to this comparison, not just excluded from it

One of tonight's seven attempts hit the Gemini API's own free-tier rate
limit (`429 RESOURCE_EXHAUSTED`, 15 requests/minute) and crashed with an
unhandled exception — before `run_smoke_test.py` ever reached its
file-write step. No trace of that attempt exists in `results/`, and
`compare_runs.py` has no way to know it happened at all; it can only
ever summarize runs that completed and wrote a file. This is accepted as
a known limitation for now (see `TECHNICAL_NOTES.md` Part 4), not a
data-quality problem with the comparison above — but worth stating
explicitly, since "6 of 7 attempts" and "6 of 6 known runs" are not the
same claim, and only the second is what §10.1's table actually shows.

### 10.6 Updated summary table

| Question | Answer, with appropriate caveats |
|---|---|
| Does §10's data finally answer §8.4/§9.3's original cross-run question? | Partially. The warning/iteration-count variation tracks *which models get proposed at all*, specifically whether Logistic Regression is tried — not the warning occurring independently of model choice. Inference from n=6, not proven. |
| Is the `ConvergenceWarning` still confirmed tied to `LogisticRegression`'s defaults, independent of dataset? | Yes, §6's original finding is unchanged and reconfirmed on 5 further runs. |
| Was the earlier same-session claim about `simplefilter("always")` protecting *across* separate runs accurate? | No — corrected in §10.3. Each `run_smoke_test.py` invocation is its own process; the real protection is against repeated warnings *within* one process, not across separately-launched runs. |
| Does the precision caveat from §8.3 still hold? | Yes — reconfirmed, with notably exact consistency (identical precision and confusion matrix) across all five recall=1.0 runs. |
| Is every run attempted this evening reflected in this comparison? | No — one failed run (API rate limit) produced no file and is invisible to `compare_runs.py` by construction; flagged as a known, unfixed limitation. |

---

<!--
DATA_SCIENCE_ANALYSIS.md update — 2026-07-27 session
-->

## 11. Update, 2026-07-27: `max_iter` exposed as a tunable hyperparameter — the ConvergenceWarning shortlist decision, tested against 3 live runs

§10's own limitations list (§10.6 / `TECHNICAL_NOTES.md` §4.5) named the
`ConvergenceWarning` itself as still unaddressed: captured and visible
since 2026-07-24, but nothing about *why* it fired had changed, and no
shortlist of options had even been drafted. This session did both —
narrowed the shortlist to a decision, implemented it
(`TECHNICAL_NOTES.md` Part 5 covers the schema/plumbing side), and
generated 3 fresh live runs to see what the agent actually does with the
new lever.

### 11.1 The shortlist, and the decision made

Three candidate approaches were discussed with Melanie:

- **(A) Raise `max_iter`'s default** (e.g. to 500) so the warning stops
  firing in most/all runs. Rejected: this would silently remove the
  exact signal §10's six-run comparison depended on — the warning's
  presence/absence is what let §10.2 distinguish "Logistic Regression
  was tried and struggled" from "Logistic Regression was never tried at
  all." Erasing the warning would have made that finding
  unreproducible going forward.
- **(B) Leave the default at 100; let the agent reason about the
  warning itself** when it appears, using `train_model`'s already-
  returned `warnings` list (§10's own capture feature) as input to its
  own decision-making.
- **(C) Expose `max_iter` as a hyperparameter the agent can choose**,
  giving it a concrete lever to act on once it notices a warning.

**Decision: B primary, with C as the concrete mechanism B needs to act
on** — not A. This preserves the warning as a real, observable event
(so §10's comparison methodology stays valid on future runs) while
giving the agent an actual way to respond to it, rather than only ever
observing it.

### 11.2 A caveat stated before, not after, seeing results

Before any run was made, it was flagged explicitly that "the data
reaching Gemini" and "Gemini reasoning well about that data" are
separate claims — a warning surfaced inside a JSON tool result is a
plumbing success regardless of whether the model does anything sensible
with it. This distinction matters for how §11.3's results below should
be read: they demonstrate the *mechanism* works, not that it will
reliably work this way on every future run.

### 11.3 Three live runs, Climate Crashes, `optimization_target="recall"`, `max_iter` now exposed

| Run | Model sequence (with `max_iter` retries shown) | Final recall / precision | Iterations | Notes |
|---|---|---|---|---|
| A | LogReg (`class_weight="balanced"`, warning fires) → LogReg retry (`max_iter=500`, same else, warning clears) → RF (`class_weight="balanced"`) → SVM (`kernel="rbf"`, `class_weight="balanced"`) | 1.0 / 0.18 | 14 | `max_iter`-only retry: metrics on the LogReg retry were **bit-identical** to the pre-retry LogReg attempt (0.787 acc / 0.25 prec / 0.778 recall, both times) — the warning cleared with zero change to predictions |
| B | LogReg (`class_weight="balanced"`, warning fires) → RF (`n_estimators=200`, `max_depth=10`, `class_weight="balanced"`) → LogReg retry (`max_iter=500`, **`C=0.1`**, `class_weight="balanced"`) | 1.0 / 0.25 | 10 | retry changed `max_iter` and `C` together |
| C | LogReg (`class_weight="balanced"`, `C=1`, warning fires) → RF (`n_estimators=200`, `class_weight="balanced"`) → LogReg retry (`max_iter=500`, **`C=0.1`**, `class_weight="balanced"`) | 1.0 / 0.25 | 10 | retry changed `max_iter` and `C` together; final metrics **bit-identical** to Run B, including the confusion matrix |

n=3, all Climate Crashes, all same `optimization_target`, generated in a
single session immediately after the schema change — a first look, not
a large sample. As with §10's six-run table, read these as observed
behavior, not a designed, controlled experiment.

### 11.4 Finding: the mechanism works, exactly as designed — documented fact, not inference

All three runs independently demonstrate the same causal chain: `train_model`
returns a `ConvergenceWarning` inside its `"warnings"` list → the agent's
own `record_convergence_decision` (Run A) or `record_model_proposal`
(Runs B, C) reasoning explicitly references it (e.g. Run A: *"the model
did not converge... I will try increasing the iterations for Logistic
Regression first"*) → the agent re-proposes Logistic Regression with a
higher `max_iter` → the warning is absent on the retry. This is the
first live confirmation that the B+C design actually functions
end-to-end, not just that the plumbing was theoretically capable of it.

### 11.5 Finding: Run A isolates `max_iter`'s own effect cleanly; Runs B/C confound it with `C`

Run A is the only one of the three where the retry changed **only**
`max_iter`, leaving every other hyperparameter unchanged. Its result is
worth stating plainly: raising `max_iter` from 100 to 500 silenced the
warning **completely**, while leaving every reported metric — accuracy,
precision, recall, F1, and the full confusion matrix — bit-identical to
the pre-retry attempt. In this run, the iteration cap was a *reporting*
problem (an honest "I can't certify convergence"), not an *accuracy*
problem; the coefficients at 100 iterations were apparently already as
good as at 500.

Runs B and C, by contrast, both changed `max_iter` **and** `C` (to 0.1)
in the same retry — the agent's own reasoning conflates the two (Run B:
*"Increasing max_iter and adjusting C might yield a more stable
result"*), so neither run can isolate which change actually drove their
jump to recall 1.0 / precision 0.25 (versus Run A's unchanged 0.778 /
0.25 after its `max_iter`-only retry). **Plausible inference, not
proven:** since Run A's `max_iter`-only change produced no metric
change at all, `C=0.1` (stronger regularization) is the more likely
driver of Runs B/C's improved precision — but this is inferred by
comparing across only 3 runs, not confirmed directly. A targeted
follow-up (propose `C=0.1` alone, `max_iter` left at its default, and
check whether recall still reaches 1.0 despite the warning still
firing) would confirm or rule this out directly, and is flagged as a
natural next data point rather than done this session.

### 11.6 Finding: a new answer that dominates the project's prior typical outcome

Runs B and C's final configuration — Logistic Regression, `C=0.1`,
`max_iter=500`, `class_weight="balanced"` — reached recall = 1.0 at
precision = 0.25. §10.1's six-run table's typical converged outcome was
SVM (`kernel="rbf"`) at the same recall (1.0) but lower precision
(0.18). This is a legitimate, citable improvement made possible
specifically by exposing `max_iter` (and, per §11.5, plausibly `C`
alongside it) — a configuration that was structurally unreachable
before this session, since `max_iter` did not previously exist as
something the agent could choose.

### 11.7 Secondary observation, flagged as curious, not explained

All three runs revisited Logistic Regression *after* trying Random
Forest — a path shape (`LR → RF → LR-retry [→ SVM]`) that did not
appear in any of §10.1's original six runs, and could not have, since
there was previously no reason to propose Logistic Regression a second
time. Separately: Run A's Random Forest step and Run C's Random Forest
step (different `n_estimators`, 100 vs. 200, both `class_weight=
"balanced"`) produced **bit-identical** metrics and confusion matrices,
despite `RandomForestClassifier`'s `random_state` never being set
anywhere in this project's schema — meaning sklearn's own internal
randomness should, in principle, differ run to run. Noted here as an
unexplained observation worth a future look, not chased down further
this session.

### 11.8 Updated summary table

| Question | Answer, with appropriate caveats |
|---|---|
| Does the agent actually use `warnings`-list data from `train_model` to change its own behavior? | Yes — confirmed directly in all 3 runs; the agent's own reasoning text references the `ConvergenceWarning` and responds by raising `max_iter` on a retry. Documented fact from live runs, not inference. |
| Does raising `max_iter` alone change the model's actual predictions? | In the one run that isolated it (Run A), no — metrics were bit-identical with and without the warning. Not yet confirmed as a general pattern; n=1 for this specific isolated comparison. |
| Is `C`, not `max_iter`, the more likely driver of Runs B/C's improved precision? | Plausible inference from comparing 3 runs, not proven — a targeted single-variable follow-up run would confirm or rule this out. |
| Does this fix outperform the project's prior typical converged outcome? | On this small sample, yes — recall 1.0 / precision 0.25 (Runs B, C) versus the prior typical 1.0 / 0.18 (§10.1). |
| Is "the data reaches Gemini" the same claim as "Gemini reasons well about it"? | No — flagged explicitly before these runs, and worth restating: this session's 3/3 success rate is encouraging but not a guarantee of future behavior on every run. |

---

<!--
DATA_SCIENCE_ANALYSIS.md update — 2026-07-29 session
-->

## 12. Update, 2026-07-29: the `max_iter`-vs-`C` confound resolved, §11.7's anomaly explained, and new Breast Cancer results

§11.5 left one thing explicitly unresolved and flagged as a natural next step: whether `C` or `max_iter` actually drove Runs B/C's improved precision. §11.7 separately flagged an odd, unexplained observation about Random Forest producing bit-identical results across runs despite `random_state` never being set. This session's real runs resolved both — and neither resolution was what the original write-up assumed.

### 12.1 Correction: the metric that actually moved was recall, not precision

Before presenting the isolation run, a correction to §11.5 and §11.6's own wording is necessary rather than quietly folded in: both sections describe Runs B/C as reaching "a better precision" (0.25) than Run A. Re-checking the actual numbers this session: **precision was 0.25 in Run A, Runs B/C, and every run discussed below — it never moved at all.** The metric that changed between Run A (`max_iter` alone) and Runs B/C (`max_iter` + `C` together) was **recall**: 0.778 vs. 1.0. This is a correction to how the finding was described, not to the finding itself — the underlying confound (which change actually drove the improvement) is unaffected, but "precision" was simply the wrong word throughout §11.5–§11.6, and is corrected here rather than left standing (see `README.md`'s "Closing the ConvergenceWarning shortlist" section for the same correction, stated more briefly).

### 12.2 The isolation run

A controlled single-variable follow-up, run this session: `C=0.1` alone, `max_iter` deliberately left at its untouched default (100) — the exact snippet flagged as a natural next step in §11.5.

| | Warning fires? | Recall | Precision |
|---|---|---|---|
| Run A (`max_iter` alone, isolated) | No | 0.778 | 0.25 |
| Runs B/C (`max_iter` + `C` together) | No | 1.0 | 0.25 |
| **Isolation run (`C` alone, this session)** | **Yes** | **1.0** | **0.25** |

The isolation run's `ConvergenceWarning` still fires — expected, since `max_iter` was deliberately untouched — yet recall still reaches 1.0, matching Runs B/C exactly. **This confirms `C`, not `max_iter`, as the driver of the recall improvement, and confirms it independently of convergence status.** `max_iter` governs only whether `lbfgs` reports having satisfied its own internal stopping criterion; it has no direct bearing on what the fitted model actually predicts. `C` (inverse regularization strength) is what reshapes the decision boundary — and the isolation run demonstrates the two are fully decoupled: a model can be "unconverged" by sklearn's own bookkeeping while making identical predictions to one that reports converging cleanly.

This upgrades §11.5's "plausible inference, not proven" to a documented, causally-isolated fact — the exact kind of controlled, single-variable-changed follow-up this project's stated portfolio priority values most (see `README.md`'s Working Preferences reference to causal-isolation analysis).

### 12.3 §11.7's anomaly, resolved: not a curiosity, a real reproducibility bug

§11.7 flagged, without explanation, that two Random Forest calls with *different* `n_estimators` values produced bit-identical metrics and confusion matrices, "despite `RandomForestClassifier`'s `random_state` never being set anywhere in this project's schema — meaning sklearn's own internal randomness should, in principle, differ run to run." This session's Breast Cancer runs surfaced the fuller picture: **three** separate Random Forest calls, each with genuinely different hyperparameters, produced identical results — then a **fourth** call, using hyperparameters matching one of the first three exactly, produced a *different* result.

Checked directly against `trainer.py`'s real code (not inferred): every estimator was instantiated as `estimator_class(**hyperparameters)`, with `random_state` never passed at all — meaning `RandomForestClassifier` always ran under sklearn's own default, `random_state=None`, drawing from numpy's global RNG, freshly (and differently) seeded in every separate process. §11.7's "bit-identical" observation was not evidence of hidden determinism; it was coincidence — this specific dataset is small and strongly separable enough that these particular hyperparameter changes (`n_estimators`, `max_depth`) often didn't move the result, until one run's random draw happened to land differently. `LogisticRegression`'s consistent reproducibility across every run in this document, by contrast, was never actually about seeding either — `lbfgs` is a deterministic optimizer on a fixed convex objective given a fixed split, with no bootstrap sampling to vary.

**Fixed this session:** `random_state` is now threaded into every estimator's constructor, reusing the same value already used for the train/test split (see `README.md`'s "Reproducibility: seeding every estimator" and `TECHNICAL_NOTES.md` Part 6 for the implementation). Verified directly: two post-fix Breast Cancer runs, same seed, different Random Forest hyperparameter combinations, now produce identical results to each other — the specific behavior the fix targets.

**Worth being precise about what this changes methodologically:** every Random Forest result reported earlier in this document (§2, §8–§11) was generated *before* this fix, under sklearn's unseeded default. Those figures remain accurate records of what those specific runs produced, but should not be read as necessarily reproducible if re-run today — a real, if narrow, caveat on this document's own earlier Random Forest figures.

### 12.4 New results: Breast Cancer across all three model types, post-fix

Three further Breast Cancer runs this session (`optimization_target="recall"`, `random_state=42` throughout) explored Logistic Regression, Random Forest, and — for the first time in this project's live-run history — an SVM with a `linear` kernel:

| Run | Model sequence | Best recall reached | Notes |
|---|---|---|---|
| 1 | RF (`class_weight="balanced"`) → SVM (`kernel="linear"`, `C=1`, `class_weight="balanced"`) | **0.9861** | Highest recall observed on Breast Cancer to date, beating every prior Logistic-Regression-based result on this dataset |
| 2 | LogReg (baseline, warning fires) → LogReg (`max_iter=500`) → RF (`n_estimators=200`, `max_depth=10`) | 0.9722 | Consistent with every earlier Logistic-Regression `max_iter=500` result on this dataset (§ Breast Cancer results throughout this document) |

The linear-kernel SVM result (Run 1) is a genuinely new data point for this project — no prior live run had proposed `kernel="linear"` specifically; every earlier SVM run used the default `rbf`. The agent's own reasoning explicitly connected the choice to the dataset's shape (*"the dataset is relatively high-dimensional (30 features) compared to the sample size, where linear models often excel"*) — a plausible, dataset-aware justification, though n=1 for this specific kernel choice, not yet a pattern.

### 12.5 Updated summary table

| Question | Answer, with appropriate caveats |
|---|---|
| Is `C`, not `max_iter`, confirmed as the driver of the recall improvement flagged in §11.5? | Yes — a controlled isolation run (`C` alone, `max_iter` untouched) reached the same recall as the confounded runs, while the warning still fired. Documented fact from a real controlled run, not inference. |
| Was §11.5/§11.6's "precision" framing accurate? | No — corrected here: precision was 0.25 throughout and never moved; recall was the metric that changed. |
| Is §11.7's Random Forest "bit-identical results" observation evidence of hidden determinism? | No — resolved as a real reproducibility bug (`random_state` never set), confirmed directly against `trainer.py`'s code and fixed this session. |
| Are pre-2026-07-29 Random Forest figures in this document still reliable as records of what happened? | Yes, as historical records of those specific runs. Not necessarily reproducible if re-run today under the old code — a narrow methodological caveat, not a retraction. |
| Has a new best recall been reached on Breast Cancer? | Yes, on this small sample — 0.9861 (linear-kernel SVM), the first live use of that kernel on this dataset. n=1, not yet a confirmed pattern. |

---

<!--
DATA_SCIENCE_ANALYSIS.md update — 2026-08-01 session
-->

## 13. Update, 2026-08-01: `summarize_run`'s final-model resolution corrected — a real breast_cancer run's reported outcome was wrong

### 13.1 What was assumed, and why it was wrong

`compare_runs.py`'s `summarize_run()` resolved a run's "final model" by taking whichever `evaluate_model` call appeared *last* in the log — reasonable in most runs, since the agent typically evaluates a model and either accepts it or proposes another. The assumption breaks whenever the agent evaluates a model purely as a *comparison point* after already having, in its own reasoning, effectively decided — exactly what real run `result_log_2026_07_29_232959_breast_cancer.json` (`optimization_target="recall"`) did.

### 13.2 The real run

Three `evaluate_model` calls, in log order:

| Order | Model | Hyperparameters | Recall |
|---|---|---|---|
| 1 | Logistic Regression | `class_weight="balanced"` | 0.9583 |
| 2 | Logistic Regression | `class_weight="balanced", max_iter=500` | **0.9722** |
| 3 | Random Forest | `n_estimators=200, max_depth=10, class_weight="balanced"` | 0.9583 |

The agent's own `record_convergence_decision` reasoning: *"The max_iter=500 logistic regression is best."* — naming the **second** evaluation, not the third (last) one. Under the old heuristic, this run's reported final model would have been the Random Forest step (recall 0.9583) — evaluated purely as a comparison point after the agent had already settled on the earlier result. Every downstream consumer of `summarize_run()`'s output (`main.py export`'s CSV rows, `main.py report`'s narrative) would have silently reported the wrong model as this run's actual outcome.

Implementation detail (the code fix itself, and the new test coverage guarding it) is a design/architecture concern, not a data-science one, per this project's own documentation split — see `TECHNICAL_NOTES.md` Part 7, §7.1–§7.2 for that side.

### 13.3 The fix, and what it changes about how to read this document

Fixed by resolving "final model" by best score on the run's own `optimization_target` rather than call order, with a new `final_model_ambiguous` flag (`True`/`False`/`None`) surfacing any disagreement so a future case like this stays visible rather than silently trusted either way (`None` covers older files with no recorded `config.target` at all — nothing to compare against, not "resolved cleanly").

**A narrow caveat on this document's own reporting practice:** figures sourced from `main.py export`/`report`'s automated output, generated before 2026-08-01, are potentially subject to this bug. The mechanism affected specifically the automated `summarize_run()` path — §2's table, for instance, predates `compare_runs.py`'s existence entirely (written 2026-07-13/17, direct log reading), so it isn't structurally exposed to this bug.

**§12.4's Run 2 entry, checked against this bug — inference, not confirmed fact:** §12.4's Run 2 (model sequence and 0.9722 recall) matches this exact run closely enough to very likely be the same one. Two things point toward that entry being unaffected by the bug, though neither is a hard confirmation: (1) `main.py export`/`report` — the CLI surface that would have exposed `summarize_run()`'s buggy output to a human — did not exist until 2026-08-01, after §12.4 was written on 2026-07-29; (2) §12.4 reports 0.9722 (the second, "actually chosen" evaluation), not 0.9583 (the third/last evaluation, which is what the old buggy heuristic would have returned). Checked against `Detailed_Session_Summary_2026-07-30.md`: it independently confirms §12.4's Run 1 figure (0.9861, linear-kernel SVM) as a real, correctly-reported result, but does not state how Run 2's figure was specifically produced. Working assumption, per Melanie's direction: §12.4's existing figures were compiled by direct reading of the raw log and the agent's reasoning text, not via the buggy automated path, and do not need retroactive correction — flagged here as an assumption, not a settled fact, should it ever need revisiting.

### 13.4 Deferred: what happens when the flag actually fires

Resolving `final_model_ambiguous=True` currently means a human reads the flag and checks the run's reasoning manually — no automated adjudication step exists yet. Folded into the still-open human-in-the-loop design conversation (items #6/#12), including the idea of asking Gemini to adjudicate a flagged mismatch via a second live call.

### 13.5 Summary table

| Question | Answer, with appropriate caveats |
|---|---|
| Was "last evaluated = final model" ever actually correct as a general assumption? | No — disproven directly by one real run, where the agent's own stated reasoning named an earlier evaluation as chosen. n=1 confirmed case, not yet surveyed across every historical run file. |
| Is this a data-science finding or a tooling bug? | Both — the cause is a summarization bug (code fix in `TECHNICAL_NOTES.md`), but its effect was a documented run's outcome being misreported, which is about trusting what this document says happened. |
| Does the fix resolve every future case automatically? | No — flags a disagreement for a human to check; automated adjudication is explicitly deferred. |
| Are any of this document's own earlier figures affected? | §12.4's Run 2 is very likely unaffected — inference from CLI-availability timing and the reported number itself, not a confirmed fact; working assumption per Melanie's direction. No other figures in this document are structurally exposed to this bug (see §13.3). |

---

This analysis is intentionally kept separate from `README.md` (which
carries only a compact summary, linking back here for full detail) and
from `TECHNICAL_NOTES.md` (which covers the architectural/implementation
side of each session's work, not live-run findings) — this file is
specifically the methodology/data-science-layer findings from real, live
agent runs.