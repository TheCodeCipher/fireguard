# FireGuard results summary (generated)

> Written by `scripts/train_all.py` (step `report`). Do not edit by hand; re-run it. File guide: `results/README.md`. Protocol: leave-one-year-out (P1), each year is tested by a model that never saw it.

## Selected model

**M3 stacked days (T=5)** (`m3_stacked_unet`), mean validation AUPRC 0.4455. Rule: highest mean validation AUPRC over the four P1 folds among M1-M4 (test split not used; ties to the smaller model).

Against the strongest soft baseline (paired event-group bootstrap, Gate 2 rule): **BEATS**.

| Model | Mean validation AUPRC |
|---|---:|
| M3 stacked days (T=5) | 0.4455 |
| M1 spatial U-Net | 0.4259 |
| M4 ConvLSTM U-Net (T=5) | 0.4169 |
| M2 attention residual U-Net | 0.4156 |

## Test metrics per fold

Pooled = all test cells of the year together; macro = mean over fires. F1/IoU use the threshold chosen on validation. Brier, BSS (Brier skill score against climatology, above 0 = better) and ECE are for the calibrated probability (Platt scaling fitted on validation); the raw-sigmoid values are in metrics/test_metrics_by_fold.csv. The BCE + Dice loss ranks cells well but pushes raw outputs away from true frequencies, which is why calibration is applied.

| Model | Fold | Pooled AUPRC | Macro AUPRC | F1 | IoU | Brier | BSS | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M1 spatial U-Net | 2018 | 0.4596 | 0.2407 | 0.5409 | 0.00194 | 0.0333 | 0.00085 |
| M1 spatial U-Net | 2019 | 0.3310 | 0.1489 | 0.4448 | 0.00060 | 0.0445 | 0.00054 |
| M1 spatial U-Net | 2020 | 0.4646 | 0.1952 | 0.5273 | 0.00161 | 0.0610 | 0.00101 |
| M1 spatial U-Net | 2021 | 0.5360 | 0.2815 | 0.5784 | 0.00115 | 0.0573 | 0.00086 |
| M2 attention residual U-Net | 2018 | 0.4719 | 0.2535 | 0.5359 | 0.00195 | 0.0273 | 0.00109 |
| M2 attention residual U-Net | 2019 | 0.3284 | 0.1466 | 0.4476 | 0.00060 | 0.0333 | 0.00053 |
| M2 attention residual U-Net | 2020 | 0.4431 | 0.1953 | 0.5241 | 0.00160 | 0.0655 | 0.00087 |
| M2 attention residual U-Net | 2021 | 0.4945 | 0.2659 | 0.5676 | 0.00118 | 0.0314 | 0.00079 |
| M3 stacked days (T=5) | 2018 | 0.4776 | 0.2372 | 0.5465 | 0.00189 | 0.0567 | 0.00098 |
| M3 stacked days (T=5) | 2019 | 0.3753 | 0.1559 | 0.4554 | 0.00058 | 0.0662 | 0.00051 |
| M3 stacked days (T=5) | 2020 | 0.4762 | 0.2022 | 0.5312 | 0.00158 | 0.0754 | 0.00098 |
| M3 stacked days (T=5) | 2021 | 0.5487 | 0.2729 | 0.5798 | 0.00112 | 0.0768 | 0.00077 |
| M4 ConvLSTM U-Net (T=5) | 2018 | 0.4526 | 0.2421 | 0.5329 | 0.00193 | 0.0366 | 0.00092 |
| M4 ConvLSTM U-Net (T=5) | 2019 | 0.3316 | 0.1458 | 0.4341 | 0.00060 | 0.0345 | 0.00051 |
| M4 ConvLSTM U-Net (T=5) | 2020 | 0.4433 | 0.1879 | 0.5213 | 0.00160 | 0.0618 | 0.00085 |
| M4 ConvLSTM U-Net (T=5) | 2021 | 0.5107 | 0.2676 | 0.5702 | 0.00116 | 0.0456 | 0.00086 |
| Distance-decay baseline | 2018 | 0.3484 | 0.1965 | 0.4769 | 0.00203 | -0.0096 | - |
| Distance-decay baseline | 2019 | 0.2256 | 0.1217 | 0.3715 | 0.00062 | 0.0077 | - |
| Distance-decay baseline | 2020 | 0.3111 | 0.1480 | 0.4481 | 0.00171 | 0.0013 | - |
| Distance-decay baseline | 2021 | 0.3959 | 0.2019 | 0.5076 | 0.00124 | -0.0162 | - |
| Climatology baseline | 2018 | 0.3390 | 0.1906 | 0.4769 | 0.00201 | 0.0000 | - |
| Climatology baseline | 2019 | 0.2162 | 0.1172 | 0.3715 | 0.00062 | 0.0000 | - |
| Climatology baseline | 2020 | 0.3021 | 0.1428 | 0.4481 | 0.00171 | 0.0000 | - |
| Climatology baseline | 2021 | 0.3868 | 0.1954 | 0.5076 | 0.00122 | 0.0000 | - |

2019 has the fewest fires (88 test events); its numbers vary most. Read every comparison per fold, not only as an average.

## Comparisons and gates

- Gate 2 (M1 vs strongest baseline): **BEATS**
- M2 attention residual U-Net vs strongest baseline: **BEATS**
- M3 stacked days (T=5) vs strongest baseline: **BEATS**
- M4 ConvLSTM U-Net (T=5) vs strongest baseline: **BEATS**
- m2_attention_res_unet vs m1_unet (residual blocks + attention gates): **REMOVE**
- m3_stacked_unet vs m1_unet (temporal context (5 days, stacked)): **REMOVE**
- m4_convlstm_unet vs m3_stacked_unet (recurrence (ConvLSTM)): **REMOVE**
- m4_convlstm_unet vs m3_stacked_unet_base48 (recurrence at matched capacity (control)): **NOT_RUN**
- m1_unet vs m1_no_terrain (terrain channels (M1 with vs without)): **NOT_RUN**
- Context 3 vs 5 days: **NOT_RUN**

Full tables with confidence intervals: `results/comparison/model_comparison.md`.

## Training

| Model | Fold | Best epoch | Epochs run | Validation AUPRC | Minutes | Parameters | Batch x accumulation | Precision |
|---|---:|---:|---:|---:|---:|---:|---|---|
| M1 spatial U-Net | 2018 | 26 | 33 | 0.3677 | 91.8 | 8.36 M | 8 x 2 | bfloat16 |
| M1 spatial U-Net | 2019 | 25 | 32 | 0.4682 | 125.6 | 8.36 M | 8 x 2 | bfloat16 |
| M1 spatial U-Net | 2020 | 23 | 30 | 0.4486 | 106.3 | 8.36 M | 8 x 2 | bfloat16 |
| M1 spatial U-Net | 2021 | 27 | 34 | 0.4190 | 130.8 | 8.36 M | 8 x 2 | bfloat16 |
| M2 attention residual U-Net | 2018 | 12 | 19 | 0.3766 | 148.1 | 8.86 M | 8 x 2 | bfloat16 |
| M2 attention residual U-Net | 2019 | 19 | 26 | 0.4675 | 560.5 | 8.86 M | 4 x 4 | bfloat16 |
| M2 attention residual U-Net | 2020 | 36 | 40 | 0.4274 | 167.9 | 8.86 M | 4 x 4 | bfloat16 |
| M2 attention residual U-Net | 2021 | 8 | 15 | 0.3908 | 70.5 | 8.86 M | 4 x 4 | bfloat16 |
| M3 stacked days (T=5) | 2018 | 17 | 24 | 0.3636 | 169.0 | 8.36 M | 16 x 1 | float16 |
| M3 stacked days (T=5) | 2019 | 25 | 32 | 0.4890 | 309.4 | 8.36 M | 16 x 1 | float16 |
| M3 stacked days (T=5) | 2020 | 20 | 27 | 0.4830 | 207.0 | 8.36 M | 16 x 1 | float16 |
| M3 stacked days (T=5) | 2021 | 24 | 31 | 0.4462 | 179.9 | 8.36 M | 16 x 1 | float16 |
| M4 ConvLSTM U-Net (T=5) | 2018 | 13 | 20 | 0.3562 | 147.6 | 17.79 M | 8 x 2 | float16 |
| M4 ConvLSTM U-Net (T=5) | 2019 | 8 | 15 | 0.4580 | 154.6 | 17.79 M | 8 x 2 | float16 |
| M4 ConvLSTM U-Net (T=5) | 2020 | 20 | 27 | 0.4484 | 219.6 | 17.79 M | 8 x 2 | float16 |
| M4 ConvLSTM U-Net (T=5) | 2021 | 9 | 16 | 0.4052 | 150.9 | 17.79 M | 8 x 2 | float16 |

Epochs are counted from 0. Training stops after 6 epochs without a validation gain and keeps the best epoch's weights.

## Figures

- `results/figures/01_test_auprc_by_fold.png`: pooled test AUPRC per fold, every model and both soft baselines
- `results/figures/02_precision_recall_curves.png`: test precision-recall curves, one panel per fold
- `results/figures/03_training_curves.png`: validation AUPRC and training loss per epoch, every model and fold
- `results/figures/04_reliability.png`: reliability of the best model's probabilities, raw and Platt-calibrated
- `results/figures/05_per_event_vs_baseline.png`: per-event test AUPRC difference to the strongest baseline
