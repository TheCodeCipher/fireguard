# Observability and label-quality diagnostics (Role 4, P3-R4-04)

**Observability (evaluation-only).** `scripts/run_observability_diagnostic.py --data-root <WSTS>`
computes, for every scored sample, the share of the TARGET day's cells whose VIIRS M11, I2 and I1
reflectance values are present, and stratifies event-macro AUPRC per model and fold into
better/poorly observed events (threshold 0.5 on the event mean, pre-registered in Phase 2). A
stratum with fewer than 10 events on a fold is reported as INSUFFICIENT.

Status: code done and tested on 4 real events; **dataset-wide run pending** (one command,
output `reports/phase3/observability_strata.csv`).

Limitation stated up front: WSTS stores reflectance composites without the VNP09GA QA bits, so this
proxy only sees *missing* reflectance (a whole day missing in one of the 4 audited events), not
cloud or smoke over a filled pixel. Expect most events in the better-observed stratum; that is a
property of the proxy, not evidence that labels are clean.

Isolation: values live in `data/evaluation_only/next_day_observability.csv`; Role 4's input
assembler rejects observability by type and name (`test_observability_not_input.py`), and a static
test fails if any training, model, inference, reader or calibration module refers to it
(`test_observability_is_evaluation_only.py`).

**Label quality (MTBS/perimeters): NOT AVAILABLE.** No MTBS or perimeter data has been obtained
or licence-checked for this project, so no label-quality audit was performed. Nothing is inferred
about perimeter truth from the VIIRS active-fire labels.
