"""Role 4 Phase 4: independent OOF validation, reveal separation and leak detection."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from fireguard_ml.evaluation.oof_validation import Approved, validate_oof
from fireguard_ml.evaluation.reveal import (
    PRE_REVEAL_SESSION_FIELDS,
    REVEAL_AUDIT_FIELDS,
    audit_record,
    find_target_leaks,
    forecast_id,
)

CH = ("a", "b")


@pytest.fixture
def world(tmp_path):
    man = tmp_path / "m.csv"
    with man.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold_test_year", "split", "group_id", "event_id", "event_year",
                    "anchor_year"])  # fmt: skip
        for fold in (2019, 2020):
            w.writerow([fold, "test" if fold == 2020 else "train", "G1", "e1", 2020, 2020])
            w.writerow([fold, "test" if fold == 2019 else "train", "G2", "e2", 2019, 2019])
    smp = tmp_path / "s.csv"
    with smp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "event_year", "target_index", "last_input_date", "target_date"])
        w.writerow(["e1", 2020, 5, "2020-08-05", "2020-08-06"])
        w.writerow(["e1", 2020, 6, "2020-08-06", "2020-08-07"])
        w.writerow(["e2", 2019, 5, "2019-07-05", "2019-07-06"])
    ok = Approved("S", "C", {"2020": "K20", "2019": "K19"}, "v1", CH)
    return man, smp, ok


def rec(**kw):
    r = {"event_id": "e1", "group_id": "G1", "fold": "2020", "origin_date": "2020-08-05",
         "target_date": "2020-08-06", "is_oof": True, "split_name": "test",
         "split_manifest_sha256": "S", "config_hash": "C", "checkpoint_sha256": "K20",
         "data_contract_version": "v1", "channels": list(CH), "is_mock": False}  # fmt: skip
    r.update(kw)
    return r


def run(tmp_path, world, recs, folds=None):
    p = tmp_path / "oof.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs))
    man, smp, ok = world
    return validate_oof(p, man, smp, ok, folds)


def test_correct_records_pass_and_missing_ones_are_listed(tmp_path, world):
    rep = run(tmp_path, world, [rec()])
    assert rep.ok and rep.accepted == 1 and rep.expected == 3
    assert set(rep.missing) == {("e1", "2020-08-06"), ("e2", "2019-07-05")}
    rep = run(tmp_path, world, [rec()], folds={2020})
    assert rep.expected == 2 and rep.missing == [("e1", "2020-08-06")]


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"fold": "2019", "checkpoint_sha256": "K19"}, "held out in fold 2020"),
        ({"group_id": "G9"}, "group id"),
        ({"split_manifest_sha256": "X"}, "split hash"),
        ({"config_hash": "X"}, "config hash"),
        ({"checkpoint_sha256": "K19"}, "checkpoint hash"),
        ({"data_contract_version": "v0"}, "data contract"),
        ({"channels": ["a", "gfs_wind_speed"]}, "channels"),
        ({"is_oof": False}, "out-of-fold"),
        ({"target_date": "2020-08-07"}, "target date"),
        ({"origin_date": "2020-08-01", "target_date": "2020-08-02"}, "scored"),
        ({"is_mock": True}, "mock"),
    ],
)
def test_each_defect_is_rejected_with_its_reason(tmp_path, world, change, reason):
    rep = run(tmp_path, world, [rec(**change)])
    assert not rep.ok and reason in rep.rejected[0][2]


def test_duplicates_are_rejected(tmp_path, world):
    assert run(tmp_path, world, [rec(), rec()]).duplicates == [("e1", "2020-08-05")]


def session():
    s = dict.fromkeys(PRE_REVEAL_SESSION_FIELDS, "x")
    s.update({"forecast_id": forecast_id("m", "e1", "2020-08-05"), "event_id": "e1",
              "origin_date": "2020-08-05", "model_version": "m", "fold": "2020",
              "target_date": "2020-08-06", "probability_artifact_uri":
              "runs/m/e1/2020-08-05/probability.tif", "fold_validation_auprc": 0.31})  # fmt: skip
    return s


def test_pre_reveal_session_has_no_target_content():
    assert find_target_leaks({"sessions": [session()]}) == []


@pytest.mark.parametrize(
    "inject",
    [
        {"actual_target_uri": "x"},
        {"target_uri": "x"},
        {"next_day_observability": 0.9},
        {"auprc_event": 0.3},
        {"n_pos": 12},
        {"f1": 0.4},
        {"note": "reveal_store/actual/abc.tif"},
        {"nested": [{"y_true": [0, 1]}]},
    ],
)
def test_leaks_are_detected_anywhere_in_a_payload(inject):
    s = session() | inject
    assert find_target_leaks({"sessions": [s]})


def test_forecast_ids_are_stable_and_distinct():
    assert forecast_id("m", "e1", "2020-08-05") == forecast_id("m", "e1", "2020-08-05")
    assert forecast_id("m", "e1", "2020-08-05") != forecast_id("m", "e1", "2020-08-06")


def test_audit_record_has_exactly_the_required_fields():
    a = audit_record("sess-1", session(), "2026-11-13T10:00:00Z")
    assert tuple(a) == REVEAL_AUDIT_FIELDS
    assert a["revealed_target_artifact_id"].startswith("actual:")


def test_reveal_package_separates_targets_from_pre_reveal(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    pytest.importorskip("fireguard_ml.data.wsts_dataset", reason="needs Role 1's reader")
    from rasterio.transform import Affine

    from fireguard_ml.evaluation.reveal import build_reveal_package

    tr = (375.0, 0.0, 600000.0, 0.0, -375.0, 4200000.0)
    data = tmp_path / "data" / "2020" / "e1"
    data.mkdir(parents=True)
    raw = np.full((23, 4, 5), np.nan, np.float32)
    raw[22, 1, 1] = 1234.0
    with rasterio.open(data / "2020-08-06.tif", "w", driver="GTiff", height=4, width=5, count=23,
                       dtype="float32", transform=Affine(*tr), crs="EPSG:32610") as d:  # fmt: skip
        d.write(raw)
    art = tmp_path / "art"
    prob = art / "runs" / "m" / "e1" / "2020-08-05" / "probability.tif"
    prob.parent.mkdir(parents=True)
    with rasterio.open(prob, "w", driver="GTiff", height=4, width=5, count=1, dtype="float32",
                       transform=Affine(*tr), crs="EPSG:32610") as d:  # fmt: skip
        d.write(np.full((1, 4, 5), 0.2, np.float32))
    oof = art / "runs" / "m" / "oof_forecasts.jsonl"
    r = rec(model_version="m", architecture="m4", protocol="P1_LOYO", horizon_hours=24,
            storage_uri="runs/m/e1/2020-08-05/probability.tif", checksum_sha256="c",
            probability_kind="model_sigmoid_uncalibrated", height=4, width=5, crs="EPSG:32610",
            transform=list(tr))  # fmt: skip
    oof.write_text(json.dumps(r))
    sel = {"fitted_on": "validation", "fold": 2020, "threshold": {"fold": 2020, "threshold": 0.3},
           "calibrator": {"fold": 2020, "a": 1.0, "b": 0.0}}  # fmt: skip
    (oof.parent / "model_version.json").write_text(json.dumps(
        {"folds": {"2020": {"selection": sel, "meta": {"val_auprc": 0.2}}}}))  # fmt: skip
    cat = oof.parent / "replay_catalogue.csv"
    cat.write_text("event_id,origin_date,artifact_verified\ne1,2020-08-05,True\n")
    out = art / "reveal_packages" / "m"
    res = build_reveal_package(cat, oof, oof.parent, tmp_path / "data", out)
    assert res["sessions"] == 1
    pre = json.loads((out / "pre_reveal" / "forecast_sessions.json").read_text())
    assert find_target_leaks(pre) == [] and tuple(pre["sessions"][0]) == PRE_REVEAL_SESSION_FIELDS
    idx = json.loads((out / "reveal_store" / "reveal_index.json").read_text())
    entry = idx[pre["sessions"][0]["forecast_id"]]
    with rasterio.open(out / entry["actual_target_uri"]) as d:
        y = d.read(1)
        assert tuple(d.transform)[:6] == tr and y.sum() == 1 and y[1, 1] == 1
    assert "reveal_store" not in json.dumps(pre)
