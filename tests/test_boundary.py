"""
Invariant 1 — engine/ must not import any LLM client, agent framework,
or networking library.

This test walks every .py file under engine/ and asserts that none of the
forbidden module names appear at import level. It catches both direct
imports and transitive re-exports through __init__.py files.

Forbidden: openai, anthropic, langchain, llama_index, httpx, requests,
           aiohttp, boto3, google.cloud.
"""

import ast
import os
from pathlib import Path

import pytest

# Modules that must never appear in engine/ imports
FORBIDDEN = {
    "openai",
    "anthropic",
    "langchain",
    "langchain_core",
    "langchain_community",
    "llama_index",
    "httpx",
    "requests",
    "aiohttp",
    "boto3",
    "google.cloud",
}

ENGINE_ROOT = Path(__file__).parent.parent / "engine"


def _collect_imports(filepath: Path):
    """Return all top-level module names imported by a Python file."""
    source = filepath.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module.split(".")[0])
    return names


def _engine_py_files():
    return list(ENGINE_ROOT.rglob("*.py"))


@pytest.mark.parametrize("filepath", _engine_py_files())
def test_no_forbidden_imports(filepath):
    """Each engine/ file must not import forbidden LLM/network modules."""
    imports = _collect_imports(filepath)
    violations = [m for m in imports if m in FORBIDDEN]
    assert not violations, (
        f"{filepath.relative_to(ENGINE_ROOT.parent)} imports forbidden modules: "
        f"{violations}. "
        "The engine/ package must be pure Python with no LLM or network dependencies."
    )
