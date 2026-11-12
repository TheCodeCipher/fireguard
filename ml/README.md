Dataset processing, models, training and evaluation

# FireGuard Machine Learning

This directory contains the deep-learning and research implementation.

## Responsibilities

- WildfireSpreadTS data loading
- Geospatial preprocessing
- U-Net spatial modelling
- ConvLSTM temporal modelling
- Experimental GNN and weather fusion
- Model training and evaluation
- Probability calibration
- Explainability

## Research Scope

The primary prediction task is next-day active-fire forecasting.

All experiments must follow the frozen event-level evaluation protocols.

## Running the tests

From the repository root:

    pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build; skip if torch is installed
    pip install -e "ml[dev]"
    cd ml && pytest

Expected: 31 passed. `ruff check src` must also be clean; `.github/workflows/ml-ci.yml` runs both.

Add `scripts` to the extras (`pip install -e "ml[dev,scripts]"`) to run anything in the repository's
`scripts/` folder, which additionally needs rasterio and matplotlib.

## Layout

The package is `fireguard_ml`, not `fireguard`: `backend/src/fireguard` already owns that name and has its
own `models` subpackage, so `from fireguard.models import ...` would resolve to the backend.

Tests live in `src/fireguard_ml/tests/` and are collected through `testpaths` in `ml/pyproject.toml`.
