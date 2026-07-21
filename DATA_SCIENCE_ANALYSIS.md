# Data Science Analysis: Agent Convergence Behavior Across Datasets

_Written 2026-07-13, following the first real end-to-end verification of
`gemini_client.py`'s agent loop. Based on 7 live runs (5 Climate Crashes, 2
Breast Cancer) against a live Gemini model. Findings below are grounded in
these actual runs — sample sizes are explicitly stated throughout, since
n=5 and n=2 are not large enough to support strong statistical claims; this
is an observational analysis of real agent behavior, not a controlled
experiment._

_Updated 2026-07-17 — see §8 for new findings from the first 3 live runs made possible by that session's orchestration-loop wiring_

_Updated 2026-07-19 — see §9: per-iteration logging — first look inside a run's actual search path_

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


This analysis is intentionally kept separate from `README.md` (which
carries only a compact summary, linking back here for full detail) and
from `TECHNICAL_NOTES.md` (which covers the architectural/implementation
side of each session's work, not live-run findings) — this file is
specifically the methodology/data-science-layer findings from real, live
agent runs.
