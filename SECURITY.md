# Security Policy

This document describes the supply chain and application security framework
applied to the `ml-agent` project. Although this project serves as a learning
exercise for the author as a solo project, these practices mitigate risks from
dependency confusion attacks, malicious package releases, silent background
upgrades, and unvalidated LLM tool-call arguments.

Adapted from the equivalent policy written for the author's prior `sql-agent`
project, with sections specific to that project's domain (SQL execution)
replaced by the actual analogous risk in this one (LLM-driven tool-call
dispatch to scikit-learn).

---

## 1. Dependency Pinning (`pyproject.toml`)

All dependencies are pinned to exact versions using a strict `uv` configuration:

```toml
[tool.uv]
add-bounds = "exact"
exclude-newer = "7 days ago"
```

### What these rules do

**`add-bounds = "exact"`** — forces `uv` to write dependencies using strict `==`
version pinning instead of loose `>=` ranges. This means no dependency can silently
upgrade to a newer version without an explicit manual decision.

**`exclude-newer = "7 days ago"`** — instructs `uv` to ignore any package release
published within the last 7 days. This creates a quarantine buffer that significantly
reduces exposure to newly injected malicious releases during their initial distribution
window, before the community has had time to detect and report them.

Because these settings live in `pyproject.toml` rather than being passed as
per-command flags, any `uv add <package>` automatically respects both rules —
no extra flags are needed to get exact pinning or the freshness bound on a
newly added dependency.

---

## 2. Dependencies

### Production

| Package | Version | Notes |
|---|---|---|
| google-genai | ==2.10.0 | Gemini API SDK — runtime AI orchestration component |
| pandas | ==3.0.3 | Dataset loading and inspection |
| python-dotenv | ==1.2.2 | Loads `GEMINI_API_KEY` from `.env` at runtime |
| scikit-learn | ==1.9.0 | Model training and evaluation |

### Development

| Package | Version | Notes |
|---|---|---|
| pytest | ==9.1.1 | Installed via the `dependency-groups.dev` group — isolated from production |
| pytest-mock | ==3.15.1 | Same group |

All versions were verified to predate the 7-day quarantine threshold before
being written to the lockfile.

---

## 3. Vulnerability Scanning

Before running any application code or the test suite, all dependencies are
scanned against the OSV vulnerability database:

```bash
uv audit
```

**Result (2026-07-13):** `Found no known vulnerabilities and no adverse project
statuses in 43 packages`

`uv audit` is an experimental `uv` feature as of this writing — pass
`--preview-features audit` to suppress the associated warning if desired.
Run whenever dependencies are added or updated, and after any `uv sync` on
a fresh clone.

An additional, opt-in malware-advisory check is also available and worth
considering as a future routine addition, though not yet adopted as a
standing practice for this project:

```bash
UV_MALWARE_CHECK=1 uv sync
```

This checks specifically against OSV's malware-advisory feed — relevant
because even after PyPI quarantines a known-malicious package, a lockfile
pointing at a direct storage URL can still install it; the standard
vulnerability audit alone doesn't cover that gap.

**Important caveat, stated plainly:** both `uv audit` and the malware check
only catch *already-known, already-reported* issues — real, useful
protection, but bounded. They don't guarantee a package is safe, only that
it isn't yet flagged.

---

## 4. Cryptographic Lockfile

`uv.lock` is committed to version control. This file contains:
- Exact versions of all direct and transitive dependencies
- Cryptographic checksums (hashes) for every package

When running `uv sync` on a fresh clone, `uv` verifies every downloaded package
against these checksums before installation. Any tampered or substituted package
will fail verification and be rejected.

This is standard, recommended practice for an application (as distinct from a
library): the lockfile contains only public package names, versions, and
hashes — no secrets or project-specific sensitive information — so committing
it is what makes the environment reproducible and cryptographically verified
on every machine that clones the repository, rather than something to
protect against exposure.

---

## 5. API Key Management

The Gemini API key is stored in a `.env` file at the project root and loaded
at runtime via `python-dotenv`'s `load_dotenv()`. This file is listed in
`.gitignore` and is never committed to version control.

A `.env.example` file is provided as a template with placeholder values.
Never hardcode API keys in source code.

**Practical note:** `load_dotenv()` must actually be called in code — having
`python-dotenv` installed does not, by itself, load `.env` into the process
environment. Tools like `uv run` do not do this automatically either;
`load_dotenv()` (or an equivalent explicit environment-loading step) is
required for the key to actually be visible to `genai.Client()` at runtime.

---

## 6. LLM-Generated Tool-Call Guard

Gemini orchestrates this project's model-search loop via native
function-calling — meaning its output determines which Python function
gets called and with what arguments. Two concrete guards are in place to
ensure this doesn't translate into arbitrary or invalid execution:

**Dispatch is restricted to an explicit whitelist.** `gemini_client.py`'s
`run_agent_loop` calls `dispatch_table[tool_name](**tool_args)`, where
`dispatch_table` contains exactly the 5 tool names defined in `tools.py`'s
`TOOL_SCHEMAS` — nothing else is reachable through this path, regardless of
what Gemini's response contains. An unrecognized `tool_name` raises a
`KeyError` rather than executing anything unintended.

**Arguments are validated against a known schema before touching
scikit-learn.** `Trainer.train_model` calls `validate_hyperparameters`,
checking `model_type` and every hyperparameter's presence, type, and
range/choice validity against `list_available_models()`'s own schema —
the same schema shown to Gemini — before any sklearn estimator is
instantiated. A hallucinated `model_type` or an out-of-range hyperparameter
raises a clear, specifically-named `ValueError` naming the exact problem,
rather than failing confusingly deep inside a third-party constructor.

```python
if not (low <= value <= high):
    raise ValueError(
        f"{key}={value!r} out of range for {model_type!r}. "
        f"Valid range: [{low}, {high}]"
    )
```

This mitigates the practical risk of trusting LLM-generated arguments
blindly — the analogous risk, in this project's domain, to `sql-agent`'s
SQL-injection guard, even though the underlying mechanism (schema
validation vs. a `SELECT`-only check) is necessarily different.

---

## 7. Safe Execution Rules

Always run code and tests inside the isolated project environment:

```bash
uv run python main.py
uv run pytest -v
```

Never use a global `python` or `pip` invocation — this bypasses the virtual
environment and the pinned dependency versions.

To clear the local package cache entirely:

```bash
uv cache clean
```

---

## 8. Reporting a Vulnerability

If you discover a security issue in this project, please open a GitHub issue
with the label `security`. For sensitive disclosures, contact the author
directly via GitHub.
