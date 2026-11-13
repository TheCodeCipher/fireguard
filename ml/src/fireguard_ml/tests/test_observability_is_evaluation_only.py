"""P3-R4-04: next-day observability can never reach model inputs, normalisation, training,
feature selection or threshold selection. Complements test_observability_not_input.py (which
guards the input assembler) with a static check of every model-side module."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
MODEL_SIDE = ["training", "models", "inference", "data/wsts_dataset.py", "data/channel_policy.py",
              "evaluation/calibration.py"]  # fmt: skip
MARKERS = ("next_day_observability", "evaluation_only", "observability")


def model_side_files():
    for rel in MODEL_SIDE:
        p = PKG / rel
        if p.is_dir():
            yield from sorted(p.glob("*.py"))
        elif p.exists():
            yield p


def code_tokens(path: Path) -> list[str]:
    """Identifiers, attributes, imports and non-docstring string literals (docs may explain)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docs.add(id(body[0].value))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.append(node.id)
        elif isinstance(node, ast.Attribute):
            out.append(node.attr)
        elif isinstance(node, ast.alias):
            out.append(node.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
        elif (
            isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs
        ):
            out.append(node.value)
    return [s.lower() for s in out]


def test_the_static_check_would_catch_a_real_reference(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        '"""observability is fine in a docstring"""\nx = load("next_day_observability")\n'
    )
    assert any("next_day_observability" in t for t in code_tokens(bad))
    ok = tmp_path / "ok.py"
    ok.write_text('"""next_day_observability mentioned in docs only"""\n')
    assert not any("observability" in t for t in code_tokens(ok))


def test_no_model_side_module_mentions_observability():
    files = list(model_side_files())
    if not files:
        pytest.skip("no model-side modules present on this branch")
    hits = [f"{f.relative_to(PKG)}: {m}" for f in files
            for m in MARKERS if any(m in tok for tok in code_tokens(f))]  # fmt: skip
    assert not hits, f"observability referenced on the model side: {hits}"


def test_observability_file_lives_outside_the_input_manifests(repo_root):
    script = repo_root / "scripts" / "run_observability_diagnostic.py"
    if not script.exists():
        pytest.skip("diagnostic script not on this branch")
    text = script.read_text(encoding="utf-8")
    assert '"evaluation_only" / "next_day_observability.csv"' in text
    assert "data/manifests/next_day_observability" not in text
