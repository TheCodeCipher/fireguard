# Documentation map

| Folder | Read it when | Main files |
|---|---|---|
| [`RUNBOOK.md`](RUNBOOK.md) | you want to run anything | setup, training, website, live forecast, checks |
| [`REPORT_GUIDE.md`](REPORT_GUIDE.md) | you write the report | what goes in each section and which file it comes from; future work |
| [`CHANGE_LOG.md`](CHANGE_LOG.md) | you want to know what changed in the November 2026 overhaul and why | grouped by topic |
| [`project/`](project/) | you need the agreed plan | `FireGuard_Final_Project_Plan_v4.md` (the plan), role execution guide |
| [`data/`](data/) | you ask "what is in the data, and what may a model use?" | `data_dictionary.md`, `leakage_audit.csv` (allowed channels, read by the code), fire-complex grouping, GFS/HLS audits, data sources and licences |
| [`evaluation/`](evaluation/) | you ask "how are models tested and compared?" | `evaluation_protocol.md`, `model_comparison_protocol.md` (gates, best-model rule), `metrics_contract.md`, `baseline_results.md`, `phase_gate_log.md` (decisions for the lead) |
| [`contracts/`](contracts/) | you change an interface between parts | API (`api_contract.md`, `openapi.json`), replay/reveal, model input and export formats, backend storage |
| [`evidence/`](evidence/) | you need the raw evidence behind a claim | JSON/CSV outputs of audits and checks (some are read by the code; do not move) |
| [`history/`](history/) | you want to know how the project got here | role hand-offs, integration log, older status reports and runbook, screenshots of the earlier site |

Generated results are not here: they are in `results/` at the repository root (`results/README.md`).

Frozen files: `configs/evaluation.yaml` and `data/manifests/*` are hashed; never edit them by hand.
