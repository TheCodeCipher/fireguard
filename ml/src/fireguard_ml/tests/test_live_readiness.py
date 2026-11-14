"""The readiness gate refuses every unsafe state of the trained artefacts (layer 1)."""

from __future__ import annotations

import json

import pytest
from live_test_helpers import CHANNELS, RUN_ID, make_repo_root, make_trained_run

from fireguard_ml.live.readiness import LiveNotReadyError, check_live_readiness

FOLDS = (2020, 2021)


def _check(tmp_path, repo=None, run_kw=None, **kw):
    run_dir, run, artifact_root = make_trained_run(tmp_path, FOLDS, **(run_kw or {}))
    repo = repo or make_repo_root(tmp_path)
    args = dict(run_dir=run_dir, run=run, folds=FOLDS, run_id=RUN_ID,
                artifact_root=artifact_root, repo_root=repo)
    args.update(kw)
    return check_live_readiness(**args), (run_dir, run, artifact_root)


def test_a_complete_gated_run_is_ready(tmp_path):
    (report, bundles), _ = _check(tmp_path)
    assert report.gate_status == "GATED" and report.gate_problems == []
    assert report.folds == [2020, 2021] and report.T == 1 and report.channels == CHANNELS
    assert report.architecture == "m1_spatial_unet" and report.is_mock is False
    assert [b.fold for b in bundles] == [2020, 2021]
    assert set(report.checkpoint_sha256_by_fold) == {"2020", "2021"}
    assert all(not b.model.training for b in bundles)
    json.dumps(report.as_dict())  # serialisable


def _problems(tmp_path, **kw):
    with pytest.raises(LiveNotReadyError) as e:
        _check(tmp_path, **kw)
    return "\n".join(e.value.problems)


def test_missing_fold_is_refused(tmp_path):
    run_dir, run, artifact_root = make_trained_run(tmp_path, (2020,))
    with pytest.raises(LiveNotReadyError, match="fold 2021: missing"):
        check_live_readiness(run_dir=run_dir, run=run, folds=FOLDS, run_id=RUN_ID,
                             artifact_root=artifact_root, repo_root=make_repo_root(tmp_path))


@pytest.mark.parametrize("sidecar", ["normalizer.json", "selection.json", "summary.json"])
def test_missing_sidecar_is_refused(tmp_path, sidecar):
    run_dir, run, artifact_root = make_trained_run(tmp_path, FOLDS)
    (run_dir / "fold_2021" / f"{run}.{sidecar}").unlink()
    with pytest.raises(LiveNotReadyError, match=sidecar):
        check_live_readiness(run_dir=run_dir, run=run, folds=FOLDS, run_id=RUN_ID,
                             artifact_root=artifact_root, repo_root=make_repo_root(tmp_path))


def test_non_strict_checkpoint_is_refused(tmp_path):
    assert "not a strict checkpoint" in _problems(tmp_path, run_kw={"strict": False})


def test_mock_checkpoint_is_refused_unless_explicitly_allowed(tmp_path):
    assert "synthetic (mock)" in _problems(tmp_path, run_kw={"is_mock": True})


def test_allow_mock_needs_mock_in_the_run_id_and_marks_the_report(tmp_path):
    run_dir, run, artifact_root = make_trained_run(tmp_path, FOLDS, is_mock=True,
                                                   run_id="m1_smoke_mock")
    repo = make_repo_root(tmp_path)
    common = dict(run_dir=run_dir, run=run, folds=FOLDS, artifact_root=artifact_root,
                  repo_root=repo, allow_mock=True)
    with pytest.raises(LiveNotReadyError, match="mock"):
        check_live_readiness(run_id="m1_real_looking", **common)
    report, _ = check_live_readiness(run_id="m1_smoke_mock", **common)
    assert report.is_mock is True


def test_changed_channel_order_is_refused(tmp_path):
    swapped = (CHANNELS[1], CHANNELS[0], *CHANNELS[2:])
    assert "channel order" in _problems(tmp_path, run_kw={"channels": swapped})


def test_other_preprocessing_version_is_refused(tmp_path):
    out = _problems(tmp_path, run_kw={"preprocessing_version": "UNSET"})
    assert "preprocessing_version" in out


def test_mixed_config_hashes_are_refused(tmp_path):
    out = _problems(tmp_path, run_kw={"config_hashes": {2020: "a" * 64, 2021: "b" * 64}})
    assert "different configs" in out


def test_checkpoint_that_differs_from_the_export_is_refused(tmp_path):
    run_dir, run, artifact_root = make_trained_run(tmp_path, FOLDS)
    mv = artifact_root / "runs" / RUN_ID / "model_version.json"
    d = json.loads(mv.read_text())
    d["folds"]["2021"]["checkpoint_sha256"] = "0" * 64
    mv.write_text(json.dumps(d))
    with pytest.raises(LiveNotReadyError, match="differs from the one exported"):
        check_live_readiness(run_dir=run_dir, run=run, folds=FOLDS, run_id=RUN_ID,
                             artifact_root=artifact_root, repo_root=make_repo_root(tmp_path),
                             allow_ungated=True)


def test_missing_export_is_refused_even_when_ungated(tmp_path):
    out = _problems(tmp_path, run_kw={"export": False}, allow_ungated=True)
    assert "model_version.json is missing" in out


def test_fold_absent_from_the_export_is_refused(tmp_path):
    run_dir, run, artifact_root = make_trained_run(tmp_path, FOLDS)
    mv = artifact_root / "runs" / RUN_ID / "model_version.json"
    d = json.loads(mv.read_text())
    del d["folds"]["2020"]
    mv.write_text(json.dumps(d))
    with pytest.raises(LiveNotReadyError, match="fold 2020: not in model_version.json"):
        check_live_readiness(run_dir=run_dir, run=run, folds=FOLDS, run_id=RUN_ID,
                             artifact_root=artifact_root, repo_root=make_repo_root(tmp_path))


def test_normaliser_of_another_fold_is_refused(tmp_path):
    run_dir, run, artifact_root = make_trained_run(tmp_path, FOLDS)
    a = run_dir / "fold_2020" / f"{run}.normalizer.json"
    (run_dir / "fold_2021" / f"{run}.normalizer.json").write_text(a.read_text())
    with pytest.raises(LiveNotReadyError, match="normaliser does not belong"):
        check_live_readiness(run_dir=run_dir, run=run, folds=FOLDS, run_id=RUN_ID,
                             artifact_root=artifact_root, repo_root=make_repo_root(tmp_path))


GATE_CASES = {
    "registry mock": (dict(run_kw={"registry_mode": "mock"}), None, "mode is not 'real'"),
    "registry missing": (dict(run_kw={"registry_mode": None}), None, "registry_bundle.json"),
    "gate2 fails": ({}, dict(gate2="FAILS"), "Gate 2 decision is 'FAILS'"),
    "gate2 not run": ({}, dict(gate2="NOT_RUN"), "Gate 2 decision is 'NOT_RUN'"),
    "gate2 missing": ({}, dict(gate2=None), "gate_decisions.json"),
    "gate2 other split": ({}, dict(gate2="BEATS", split="f" * 64), "different split manifest"),
}


@pytest.mark.parametrize("case", sorted(GATE_CASES))
def test_gate_failures_refuse_unless_allow_ungated_which_stamps_the_output(tmp_path, case):
    kw, repo_kw, text = GATE_CASES[case]
    repo = make_repo_root(tmp_path, **repo_kw) if repo_kw is not None else None
    run_dir, run, artifact_root = make_trained_run(tmp_path, FOLDS, **kw.get("run_kw", {}))
    repo = repo or make_repo_root(tmp_path)
    args = dict(run_dir=run_dir, run=run, folds=FOLDS, run_id=RUN_ID,
                artifact_root=artifact_root, repo_root=repo)
    with pytest.raises(LiveNotReadyError) as e:
        check_live_readiness(**args)
    assert text in "\n".join(e.value.problems)
    report, bundles = check_live_readiness(**args, allow_ungated=True)
    assert report.gate_status == "UNGATED" and any(text in p for p in report.gate_problems)
    assert len(bundles) == 2


def test_matches_is_a_passing_gate(tmp_path):
    (report, _), _ = _check(tmp_path, repo=make_repo_root(tmp_path, gate2="MATCHES"))
    assert report.gate_status == "GATED"


def test_readiness_reads_only(tmp_path):
    run_dir, run, artifact_root = make_trained_run(tmp_path, FOLDS)
    repo = make_repo_root(tmp_path)
    before = {p: (p.stat().st_mtime_ns, p.stat().st_size)
              for p in tmp_path.rglob("*") if p.is_file()}
    check_live_readiness(run_dir=run_dir, run=run, folds=FOLDS, run_id=RUN_ID,
                         artifact_root=artifact_root, repo_root=repo)
    after = {p: (p.stat().st_mtime_ns, p.stat().st_size)
             for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
