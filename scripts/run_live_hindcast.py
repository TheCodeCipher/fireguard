"""Layer-4 check: does the live input path keep the model's skill on held-out WSTS events?

    python scripts/run_live_hindcast.py --data-root $WSTS --run-dir outputs/m1 \
        --run m1_spatial_unet_T1_seed0 --run-id m1_t1_v1 --artifact-root artifacts

Run after the eight training commands. For test events of every fold, the forecast is made twice
with the checkpoint of the fold that HELD THE EVENT OUT (``assert_held_out``), never with the
fold average, so nothing seen in training inflates the result:

1. inputs from the WSTS files (``WSTSTemporalSource``) - what Gate 2 measured;
2. inputs rebuilt by the live recipe (FIRMS archive fire, Earth Engine terrain and reflectance,
   on the event's own grid).

Both are scored against the same WSTS target. Pass: pooled AUPRC on rebuilt inputs within
``--max-rel-drop`` (10%) of the AUPRC on WSTS inputs. Dilated persistence of the rebuilt fire is
reported as a reference (F1 / IoU; it is binary, so it has no meaningful AUPRC). CPU by default.
Writes only ``docs/evidence/live/live_hindcast.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import yaml

from fireguard_ml.baselines.persistence import dilated_persistence, fire_mask
from fireguard_ml.data.event_splits import fold_events, read_manifest
from fireguard_ml.data.wsts_dataset import ACTIVE_FIRE_BAND, WSTSTemporalSource
from fireguard_ml.evaluation.metrics import (
    counts_by_score,
    f1_iou,
    f1_iou_from_confusion,
    merge_counts,
    summarise,
)
from fireguard_ml.inference.temporal_oof import assert_held_out
from fireguard_ml.live.active_fire import DetectionRasterizer, FirmsAreaProvider
from fireguard_ml.live.earth_engine import EarthEngineTerrain, EarthEngineVNP09GA
from fireguard_ml.live.forecast import predict_fold
from fireguard_ml.live.frames import LiveTemporalSource
from fireguard_ml.live.grid import region_from_grid
from fireguard_ml.live.readiness import LiveNotReadyError, check_live_readiness
from fireguard_ml.live.reflectance import LinearReflectanceMapping

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[2018, 2019, 2020, 2021])
    ap.add_argument("--events-per-fold", type=int, default=10)
    ap.add_argument("--reflectance", choices=("v1", "v2"), default="v2",
                    help="v1 = the collection WSTS used (isolates fire/terrain effects)")
    ap.add_argument("--max-rel-drop", type=float, default=0.10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "evidence" / "live")
    a = ap.parse_args()

    try:  # hard checks only: the hindcast is part of deciding whether to ungate
        report, bundles = check_live_readiness(
            run_dir=a.run_dir, run=a.run, folds=a.folds, run_id=a.run_id,
            artifact_root=a.artifact_root, repo_root=ROOT, allow_ungated=True)
    except LiveNotReadyError as exc:
        raise SystemExit(f"NOT READY: {exc}") from None
    live_cfg = yaml.safe_load((ROOT / "configs" / "live_forecast.yaml").read_text())
    af = live_cfg["active_fire"]
    mp = ROOT / live_cfg["reflectance"]["mapping_path"]
    mapping = LinearReflectanceMapping.from_json(mp) if mp.is_file() else None
    refl = EarthEngineVNP09GA(a.reflectance, mapping)
    terrain = EarthEngineTerrain()
    firms = FirmsAreaProvider(os.environ.get("FIRMS_MAP_KEY", ""), "VIIRS_SNPP_SP")
    rows = read_manifest(ROOT / "data" / "manifests" / "p1_loyo_fold_manifest.csv")
    samples_csv = ROOT / "data" / "manifests" / "wsts_eval_samples.csv"
    dev = torch.device(a.device)
    rng = random.Random(a.seed)

    parts = {"wsts": [], "live": []}
    pers_conf = np.zeros(3, dtype=np.int64)
    abs_diff, per_event = [], []
    for b in bundles:
        test = fold_events(rows, b.fold, "test")
        wsts = WSTSTemporalSource(a.data_root, report.channels, report.T, samples_csv, test)
        events = sorted({k[0] for k in wsts.keys()})
        for ev in rng.sample(events, min(a.events_per_fold, len(events))):
            assert_held_out(ev, b.fold, rows)
            keys = [k for k in wsts.keys() if k[0] == ev]
            s0 = wsts.load(keys[0])
            region = region_from_grid(ev, s0.transform, *s0.hw, s0.crs or "EPSG:32610")
            first = min(k[1] for k in keys) - timedelta(days=report.T)
            dets = firms.detections(region.bbox_wgs84, first - timedelta(days=1),
                                    max(k[1] for k in keys))
            fire = DetectionRasterizer(
                dets, firms.source, exclude_confidence=tuple(af["exclude_confidence"]),
                buffer_m=float(af["buffer_m"]), rule=af["rasterize_rule"],
                utc_offset_hours=float(af["day_utc_offset_hours"]))
            ev_parts = {"wsts": [], "live": []}
            for key in keys:
                ws = wsts.load(key)
                origin = key[1] - timedelta(days=1)
                live = LiveTemporalSource([region], origin, report.channels, report.T, fire,
                                          refl, terrain)
                ls = live.load((ev, key[1]))
                p_w = predict_fold(ws, b, dev)
                p_l = predict_fold(ls, b, dev)
                y, valid = ws.y.astype(bool), ws.valid & np.isfinite(p_w) & np.isfinite(p_l)
                for name, p in (("wsts", p_w), ("live", p_l)):
                    c = counts_by_score(y, np.where(valid, p, 0.0), valid)
                    parts[name].append(c)
                    ev_parts[name].append(c)
                abs_diff.append(float(np.nanmean(np.abs(p_w - p_l)[valid])))
                origin_fire = live.raw_frame(ev, origin)[ACTIVE_FIRE_BAND]  # as Role 4's baselines
                ref = dilated_persistence(fire_mask(origin_fire), 0.375)
                pi = f1_iou(y, ref, valid)
                pers_conf += [pi["tp"], pi["fp"], pi["fn"]]
            per_event.append({"event_id": ev, "fold": b.fold, "days": len(keys),
                              **{f"auprc_{n}": summarise(merge_counts(ev_parts[n]))["auprc"]
                                 for n in ev_parts}})
            print(f"fold {b.fold} {ev}: {len(keys)} days", flush=True)

    m_w, m_l = (summarise(merge_counts(parts[n])) for n in ("wsts", "live"))
    rel_drop = (m_w["auprc"] - m_l["auprc"]) / max(m_w["auprc"], 1e-12)
    result = {"run_id": a.run_id, "reflectance": refl.name, "folds": a.folds,
              "n_events": len(per_event), "n_days": len(abs_diff),
              "pooled_auprc_wsts_inputs": m_w["auprc"], "pooled_auprc_live_inputs": m_l["auprc"],
              "relative_drop": rel_drop, "max_relative_drop": a.max_rel_drop,
              "mean_abs_probability_difference": float(np.mean(abs_diff)) if abs_diff else None,
              "brier_wsts_inputs": m_w["brier"], "brier_live_inputs": m_l["brier"],
              "dilated_persistence_live_inputs": f1_iou_from_confusion(*map(int, pers_conf)),
              "per_event": per_event,
              "status": "PASS" if rel_drop <= a.max_rel_drop else "FAIL"}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "live_hindcast.json").write_text(json.dumps(result, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_event"}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
