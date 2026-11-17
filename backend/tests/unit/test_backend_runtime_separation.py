from pathlib import Path


def test_backend_runtime_dependencies_do_not_include_ml_frameworks():
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8").lower()
    dependencies = content.split("[project.optional-dependencies]", maxsplit=1)[0]
    assert "torch" not in dependencies
    assert "torchvision" not in dependencies
    assert "cuda" not in dependencies
