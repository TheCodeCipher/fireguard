# Calibration report (generated)

> `scripts/run_calibration_report.py`. Calibrators and thresholds are fitted on each fold's validation groups at natural prevalence and applied only to that fold.

Uncertainty: **no uncertainty layer exists.** FireGuard outputs one calibrated or uncalibrated probability per cell; there is no ensemble spread or predictive-interval product. A probability is not an uncertainty estimate of itself.

| Fold | Model | Probabilities | Test ECE (15 equal-mass bins) | Platt a | Platt b |
|---:|---|---|---:|---:|---:|
| 2018 | distance_decay | raw | 0.00086 | 0.561 | -0.509 |
| 2018 | distance_decay | platt_validation_fitted | 0.00019 | 0.561 | -0.509 |
| 2018 | climatology | raw | 0.00017 | 0.927 | -0.198 |
| 2018 | climatology | platt_validation_fitted | 0.00025 | 0.927 | -0.198 |
| 2019 | distance_decay | raw | 0.00015 | 0.591 | -0.507 |
| 2019 | distance_decay | platt_validation_fitted | 0.00013 | 0.591 | -0.507 |
| 2019 | climatology | raw | 0.00022 | 1.012 | -0.079 |
| 2019 | climatology | platt_validation_fitted | 0.00013 | 1.012 | -0.079 |
| 2020 | distance_decay | raw | 0.00055 | 0.578 | -0.470 |
| 2020 | distance_decay | platt_validation_fitted | 0.00005 | 0.578 | -0.470 |
| 2020 | climatology | raw | 0.00006 | 0.946 | -0.120 |
| 2020 | climatology | platt_validation_fitted | 0.00005 | 0.946 | -0.120 |
| 2021 | distance_decay | raw | 0.00049 | 0.567 | -0.509 |
| 2021 | distance_decay | platt_validation_fitted | 0.00004 | 0.567 | -0.509 |
| 2021 | climatology | raw | 0.00015 | 0.975 | -0.158 |
| 2021 | climatology | platt_validation_fitted | 0.00015 | 0.975 | -0.158 |

Reliability tables (mean predicted vs observed per bin): `reports/phase3/reliability_tables.csv`.
