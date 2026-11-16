"""Build the code package for training on Kaggle (upload it as a Kaggle dataset).

    python scripts/kaggle/build_package.py      ->  outputs/kaggle/fireguard_kaggle_code.zip

The zip holds kaggle_train.py at its top level plus the part of Prototype/ that training needs
(ml/src, configs, data/manifests, the channel audit, scripts/train_all.py), about 3 MB. The array
cache is uploaded separately (docs/RUNBOOK.md, "Training on Kaggle").
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "kaggle" / "fireguard_kaggle_code.zip"
INCLUDE = [
    "ml/pyproject.toml",
    "ml/src",
    "configs",
    "data",
    "docs/data/leakage_audit.csv",
    "scripts/train_all.py",
]
SKIP_PARTS = {"__pycache__", "tests", ".pytest_cache"}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(Path(__file__).with_name("kaggle_train.py"), "kaggle_train.py")
        for item in INCLUDE:
            p = ROOT / item
            files = (
                [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())
            )
            for f in files:
                rel = f.relative_to(ROOT)
                if (
                    SKIP_PARTS & set(rel.parts)
                    or f.suffix == ".pyc"
                    or ".egg-info" in str(rel)
                ):
                    continue
                z.write(f, Path("Prototype") / rel)
                n += 1
    print(f"wrote {OUT} ({n} files, {OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
