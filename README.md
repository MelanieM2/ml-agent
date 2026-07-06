# ml-agent

Agentic ML experimentation assistant — an LLM (Gemini) orchestrates dataset inspection, model proposal, training, and evaluation over scikit-learn via native function-calling, iterating toward a target metric in a supervised loop.

**Status: in progress.** This is Project 5 in a 10-step self-directed learning roadmap toward Data Science / ML Engineering / Agentic AI. `dataset.py` is complete and tested; `tools.py`, `gemini_client.py`, `trainer.py`, and `agent.py` are in progress.

---

## Concept

Instead of manually writing and tuning scikit-learn models, an LLM orchestrates the pipeline:

```
User: "Find the best classifier for this dataset, optimizing for recall"
        ↓
Agent inspects the dataset (via Pandas) — shape, types, target distribution
        ↓
Agent (via Gemini, native function-calling) proposes candidate models + reasoning
        ↓
Agent trains each via scikit-learn, computes metrics
        ↓
Agent (via Gemini) interprets results, decides: good enough, or iterate further?
        ↓
Loop until convergence or max iterations
        ↓
Agent reports: best model, why, plain-English summary
```

This extends the agentic loop pattern from a prior project, `sql-agent` (question → LLM reasoning → execution → interpretation → loop), into a second domain, using **native Gemini function-calling** rather than the simpler prompt-schema approach used previously — a deliberate step up in tool-calling sophistication.

### Core architectural principle

Deterministic code handles fact-gathering; the LLM handles reasoning under uncertainty. `dataset.py`'s `inspect_dataset()` computes shape, class balance, missing values, and feature statistics — but makes no judgment calls. Deciding what those facts *imply* (e.g. "this class imbalance means recall-aware modeling is appropriate") is left to Gemini, reasoning over the structured facts it's handed. The project does not automate data-science judgment — it automates the mechanical prep work so that judgment has good inputs to work from.

---

## Project structure

```
ml-agent/
├── pyproject.toml
├── .env.example
├── .gitignore
├── AGENTS.md              # agent rules, guardrails, tool list, convergence criteria
├── main.py                # CLI entry point / interactive loop
├── ml_agent/
│   ├── dataset.py         # dataset-agnostic loading + inspection ✅ done
│   ├── tools.py            # Gemini function-calling tool schemas — next up
│   ├── gemini_client.py    # Gemini client wrapper + function-call dispatch
│   ├── trainer.py          # scikit-learn training + metric computation
│   └── agent.py             # orchestration: state, convergence logic, main loop
└── tests/
    ├── test_dataset.py    # ✅ done, 6 passing tests
    ├── test_tools.py
    └── test_trainer.py
```

### Why `agent.py` and `gemini_client.py` are separate
`gemini_client.py` only knows how to talk to Gemini and dispatch function calls. `agent.py` owns the actual multi-step state — what's been tried, results so far, whether to keep iterating. This separation allows convergence logic to be unit-tested without a live API call.

---

## Datasets

- **Primary: Climate Model Simulation Crashes** (OpenML id 1467). Binary classification — predict whether a given combination of 18 physical parameters (from the POP2 ocean model component of CCSM4) causes a numerical simulation crash. 540 rows, genuinely imbalanced (~91.5% success / 8.5% failure), which gives a "optimize for recall" objective real substance. Loader explicitly drops two non-predictive identifier columns present in the raw source (documented by the dataset's own maintainers as unfit for prediction) and remaps OpenML's string target labels to standard `0`/`1` semantics, with `1` established as the failure/positive class via cross-checking against the dataset's documented failure count.
- **Fallback: Breast Cancer Wisconsin** (`sklearn.datasets.load_breast_cancer`). Used for fast, network-free debugging of the pipeline in isolation. Note: its positive-class convention (`1 = benign`) is the *opposite* sense from Climate Crashes (`1 = failure`) — `pos_label` is established per-dataset, never assumed consistent.

Datasets are loaded through a small registry (`DATASET_LOADERS`) in `dataset.py`, keyed by short name (`"climate"`, `"breast_cancer"`), so adding a new dataset later requires only a new loader function and one registry line.

---

## Setup

```bash
git clone git@github.com:MelanieM2/ml-agent.git
cd ml-agent
uv sync
cp .env.example .env   # then fill in your real GEMINI_API_KEY
```

Dependencies are pinned exactly (`add-bounds = "exact"` in `[tool.uv]`) and restricted to packages no newer than 7 days old at install time (`exclude-newer = "7 days ago"`), as a routine security precaution applied to every project in this series.

## Running

```bash
uv run python main.py
```

(Interactive loop entry point — details to be filled in as `agent.py` is completed.)

## Testing

```bash
uv run pytest -v
```

Network-dependent behavior (the Climate Crashes OpenML fetch) is not exercised directly in tests — it's mocked, keeping the suite fast and independent of external service availability, consistent with how Gemini API calls are mocked in this project's `sql-agent` predecessor.

---

## Security

- `.env` holds the real `GEMINI_API_KEY`; it is gitignored from project creation, before any commit, so the key never enters git history.
- `.env.example` is committed as a placeholder template.
- Dependency versions are exact-pinned and freshness-bounded (see `pyproject.toml`).

---

## Development Notes

This project was developed with AI assistance:
- **Development tool:** Claude Sonnet 5 (Anthropic), used as a pair-programming and design-review collaborator — explaining rationale, reviewing architecture decisions, and catching data-quality issues (e.g. identifying and correcting two non-predictive identifier columns and a target-label encoding ambiguity in the Climate Crashes dataset) before they reached the codebase.
- **Runtime AI component:** `gemini-3.1-flash-lite` (Google), the model actually orchestrating the agent's tool-calling loop at runtime — this is the AI system the project *is*, as distinct from the AI assistance used to *build* it.

All architectural decisions, dataset choices, and correctness checks (e.g. cross-referencing dataset documentation against observed data) were reviewed and made by the project author. AI assistance focused on explaining concepts, generating boilerplate, and surfacing issues for human review — not on unsupervised code generation.

---

## Roadmap context

Project 5 of a 10-step learning roadmap: Linux → Git → Professional Python → Pandas → SQL → **Scikit-learn (this project)** → AI-assisted workflows → PyTorch → Transformers → Geometric Deep Learning. Builds directly on the agentic-loop pattern established in `sql-agent` (Project 3). Grokking and Geometric Deep Learning are deliberately deferred to later projects; this project's explicit focus is agentic/software engineering skill, not ML theory.
