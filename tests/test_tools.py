"""Tests for tools.py: catches drift between TOOL_SCHEMAS and the real
tool function signatures.

Convention: keyword-only
parameters are values injected internally via functools.partial by
whatever wires the dispatch table (agent.py) - never something Gemini
supplies or should see in a schema. This test excludes keyword-only
parameters from the comparison; only positional-or-keyword parameters
are checked against each TOOL_SCHEMAS entry's declared properties.
"""
from __future__ import annotations

import inspect

import pytest

from ml_agent.tools import (
    TOOL_SCHEMAS,
    TOOL_FUNCTIONS,
    evaluate_model,
    list_available_models,
    record_convergence_decision,
    record_model_proposal,
    train_model,
)

# Maps each schema's "name" to its real callable, for lookup by name.
# TOOL_FUNCTIONS = {
#     "list_available_models": list_available_models,
#     "train_model": train_model,
#     "evaluate_model": evaluate_model,
#     "record_model_proposal": record_model_proposal,
#     "record_convergence_decision": record_convergence_decision,
# }


def _gemini_visible_params(fn):
    """Returns only the parameters Gemini is meant to see/supply - i.e.
    excludes keyword-only params, which are internally injected via
    functools.partial (see evaluate_model's docstring in tools.py)."""
    sig = inspect.signature(fn)
    return {
        name: param
        for name, param in sig.parameters.items()
        if param.kind != inspect.Parameter.KEYWORD_ONLY
    }


@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda s: s["name"])
def test_schema_param_names_match_function(schema):
    """Every param name in the function must appear in the schema, and
    vice versa - catches renamed/added/removed arguments on either side."""
    name = schema["name"]
    real_params = _gemini_visible_params(TOOL_FUNCTIONS[name])
    schema_params = schema["parameters"]["properties"]

    real_names = set(real_params)
    schema_names = set(schema_params)

    missing_from_schema = real_names - schema_names
    extra_in_schema = schema_names - real_names

    assert not missing_from_schema, (
        f"{name}: function has parameter(s) {missing_from_schema} "
        f"not declared in TOOL_SCHEMAS"
    )
    assert not extra_in_schema, (
        f"{name}: TOOL_SCHEMAS declares parameter(s) {extra_in_schema} "
        f"the function doesn't accept"
    )


@pytest.mark.parametrize("schema", TOOL_SCHEMAS, ids=lambda s: s["name"])
def test_required_params_match_defaults(schema):
    """Every schema 'required' entry must correspond to a real parameter
    with no default - and every parameter with no default must be listed
    as required. Catches e.g. a newly-optional arg whose schema entry
    was never updated to move it out of 'required'."""
    name = schema["name"]
    real_params = _gemini_visible_params(TOOL_FUNCTIONS[name])

    required_in_schema = set(schema["parameters"].get("required", []))
    required_in_function = {
        pname
        for pname, param in real_params.items()
        if param.default is inspect.Parameter.empty
    }

    assert required_in_schema == required_in_function, (
        f"{name}: schema 'required' {required_in_schema} does not match "
        f"the function's actually-required params {required_in_function}"
    )


def test_every_schema_has_a_registered_function():
    """Guards the two lists themselves from drifting apart - e.g. a new
    tool added to TOOL_SCHEMAS but never wired into TOOL_FUNCTIONS above,
    or vice versa (a callable Gemini could never actually reach)."""
    assert {s["name"] for s in TOOL_SCHEMAS} == set(TOOL_FUNCTIONS)


def test_dispatch_table_override_keys_exist():
    """agent.py's build_dispatch_table overrides exactly two TOOL_FUNCTIONS
    entries by literal string key:
        {**TOOL_FUNCTIONS, "train_model": bound_train,
         "evaluate_model": bound_evaluate}
    This doesn't import agent.py (keeps this file's dependencies as they
    are) - it just guards the two key names agent.py's override relies on
    existing here, so a rename in tools.py surfaces at test time instead
    of silently at the next live run."""
    assert "train_model" in TOOL_FUNCTIONS
    assert "evaluate_model" in TOOL_FUNCTIONS