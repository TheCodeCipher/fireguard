# fireguard

Multimodal spatio-temporal deep learning platform for wildfire intelligence, probabilistic forecasting and historical fire replay.

# 🔥 FireGuard

### Multimodal Wildfire Intelligence and Probabilistic Forecasting Platform

**FireGuard** is a research-oriented wildfire intelligence platform that combines satellite observations, environmental data and deep learning to investigate and visualize the probability of future wildfire activity.

The project aims to go beyond conventional active-fire monitoring by developing a multimodal spatio-temporal forecasting model and an interactive historical forecasting and replay environment.

> **Project Status:** Research prototype under development.

---

## 1. Project Overview

Wildfires evolve through complex interactions between vegetation, weather, terrain and existing fire activity. Monitoring active fires alone does not provide an understanding of how wildfire activity may evolve.

FireGuard investigates whether multimodal deep learning can improve the forecasting of newly active-fire detections over a 24-hour horizon.

The initial research implementation uses the **WildfireSpreadTS benchmark**, which contains large wildfire events from the western United States.

The primary task is next-day active-fire prediction, with newly detected active-fire cells evaluated as a secondary research metric.

## 2. Core Objectives

- Integrate satellite-derived active-fire observations.
- Develop a spatial wildfire forecasting model.
- Investigate temporal deep learning for wildfire evolution.
- Combine fire history, weather, terrain and vegetation information.
- Generate probabilistic spatial forecasts.
- Evaluate prediction uncertainty and calibration.
- Measure generalization across unseen wildfire events and years.
- Provide an interactive historical prediction-versus-observation simulator.

## 3. Proposed Deep Learning Architecture

| Component            | Architecture                    | Purpose                               |
| -------------------- | ------------------------------- | ------------------------------------- |
| Spatial modelling    | U-Net / CNN                     | Spatial feature extraction            |
| Temporal modelling   | ConvLSTM                        | Temporal wildfire evolution           |
| Weather conditioning | MLP + FiLM                      | Experimental weather fusion           |
| Graph modelling      | GCN / GAT                       | Experimental geographic relationships |
| Calibration          | Temperature scaling             | Probability calibration               |
| Explainability       | Integrated Gradients / Grad-CAM | Prediction interpretation             |

The strict default architecture excludes GFS channels. GNN, HLS model integration and other optional components will only be retained if they pass feasibility and evaluation gates.

## 4. Dataset Sources

| Dataset                    | Purpose                                        |
| -------------------------- | ---------------------------------------------- |
| WildfireSpreadTS           | Primary training and evaluation benchmark      |
| NASA FIRMS VIIRS           | Active-fire observation and historical context |
| GRIDMET                    | Historical weather and drought-related inputs  |
| GFS                        | Experimental forecast-weather features         |
| SRTM                       | Terrain information                            |
| MODIS Land Cover           | Land-cover information                         |
| NASA HLS S30 v2.0          | Optional spectral feature feasibility study    |
| VNP09GA                    | Evaluation-only observability diagnostic       |
| MTBS / Landsat Burned Area | Event-level label-quality validation           |
| LANDFIRE                   | Optional additional fuel information           |

Raw datasets will remain outside the Git repository. Refer to `docs/data/` for dataset documentation and access information.

## 5. Research Scope

**Primary research task:** Predict the next-day spatial distribution of VIIRS-detected active fire.

**Secondary research task:** Evaluate performance on newly detected active-fire cells.

### Research Question

Can multimodal spatial and temporal deep learning improve calibrated 24-hour wildfire forecasts over strong non-deep-learning baselines under unseen-event and unseen-year evaluation?

### Evaluation Metrics

- AUPRC (primary metric)
- F1-score and IoU
- Brier Skill Score
- Expected Calibration Error
- Reliability analysis
- Event-level paired bootstrap confidence intervals
- Temporal and geographic generalization
- Robustness to missing optical observations

## 6. Application Modules

### Live Monitor

Displays active-fire observations from NASA FIRMS, including location, timestamp, satellite source and available detection attributes.

This is an observation-only module in v1.

### Forecast & Replay Lab

Allows users to select historical wildfire events and forecast origins, retrieve precomputed out-of-fold predictions, and compare forecasts against subsequent satellite observations.

### Explainability and Uncertainty

Displays prediction probabilities, uncertainty information and feature-level explanations.

### Methodology and Data

Documents model architecture, dataset sources, limitations and evaluation methodology.

### Citizen Reporting

A prototype interface for submitting fire/smoke photographs, timestamps and locations for human review.

## 7. Technology Stack

| Layer                 | Technology              |
| --------------------- | ----------------------- |
| Deep learning         | PyTorch                 |
| Backend               | Python, FastAPI         |
| Validation            | Pydantic                |
| Database              | PostgreSQL              |
| ORM                   | SQLAlchemy              |
| Frontend              | React, TypeScript, Vite |
| Styling               | Tailwind CSS            |
| Maps                  | MapLibre GL JS          |
| Geospatial processing | Rasterio, GeoPandas     |
| Testing               | Pytest, HTTPX           |
| Version control       | Git, GitHub             |
| CI                    | GitHub Actions          |
| Deployment            | Docker                  |

## 8. Repository Structure

```text
fireguard/
├── .github/
├── backend/
│   ├── src/fireguard/
│   └── tests/
├── ml/
│   ├── src/fireguard_ml/
│   └── tests/
├── frontend/
│   └── src/
├── configs/
├── data/
├── docs/
├── scripts/
├── tests/
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
└── README.md
```

## 9. Development Phases

| Phase   | Focus                                                     |
| ------- | --------------------------------------------------------- |
| Phase 1 | Data feasibility, leakage audit and repository foundation |
| Phase 2 | Baselines, WSTS loader and U-Net                          |
| Phase 3 | Temporal modelling and multimodal experiments             |
| Phase 4 | Backend, frontend and historical replay integration       |
| Phase 5 | Evaluation, calibration, testing and final demonstration  |

Refer to `docs/project/` for the detailed project plan and role-wise execution guide.

## 10. Development Workflow

FireGuard follows a branch-based collaborative workflow.

- `main` contains approved and integrated code.
- Every task must have a GitHub Issue.
- Contributors work on role-specific feature branches.
- Direct updates to `main` are prohibited.
- Changes must be submitted through Pull Requests.
- Pull Requests require review and acceptance testing.
- Approved contributions are merged into `main`.

Example branches:

```text
role-1/data/wsts-loader
role-2/ml/unet-baseline
role-3/ml/convlstm
role-4/research/evaluation
role-5/backend/api-foundation
role-6/frontend/live-monitor
```

## 11. Data and Security Policy

- Raw datasets must never be committed to GitHub.
- Credentials and API keys must be stored in environment variables.
- `.env` files must never be committed.
- Dataset redistribution must follow the original dataset license.
- Uploaded user reports must be handled securely.
- Predictions must include their provenance and limitations.
- FireGuard is a research decision-support prototype, not an official emergency-warning or evacuation authority.

## 12. Project Documentation

- `docs/architecture/` – System and model architecture.
- `docs/data/` – Dataset inventory, data dictionaries and leakage audit.
- `docs/api/` – Backend API contracts.
- `docs/experiments/` – Experiments and evaluation results.
- `docs/project/` – Project plan and execution documentation.

---

**FireGuard – From satellite observations to explainable wildfire forecasting research.**
