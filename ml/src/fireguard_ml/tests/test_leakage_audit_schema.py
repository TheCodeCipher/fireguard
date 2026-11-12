"""Schema tests for docs/leakage_audit.csv. CI fails if a malformed row is introduced."""

from __future__ import annotations

import csv

import pytest

from fireguard_ml.data.channel_policy import (
    INPUT_DECISIONS,
    REGISTER_COLUMNS,
    TIME_STATUSES,
    load_register,
    policy_violations,
)

EXPECTED_CHANNELS = 24  # 23 raw WSTS channels (land cover as one row) + derived binary fire mask


@pytest.fixture(scope="module")
def register_path(repo_root):
    return repo_root / "docs" / "leakage_audit.csv"


def test_header_matches_the_contract_exactly(register_path):
    with register_path.open(newline="", encoding="utf-8-sig") as f:
        assert tuple(next(csv.reader(f))) == REGISTER_COLUMNS


def test_every_row_has_valid_statuses_decision_and_evidence(register_path):
    reg = load_register(register_path)
    assert len(reg) == EXPECTED_CHANNELS
    for r in reg.values():
        assert r.no_lookahead in TIME_STATUSES
        assert r.operationally_available in TIME_STATUSES
        assert r.input_decision in INPUT_DECISIONS
        assert r.evidence.strip()


def test_no_pass_without_a_named_evidence_artifact(register_path):
    # a PASS must cite code (".py"), a product/doc source, or a dated Role decision
    for r in load_register(register_path).values():
        if "PASS" in (r.no_lookahead, r.operationally_available):
            assert any(k in r.evidence for k in (".py", "documentation", "Role 4 decision")), r.name


def test_register_is_internally_consistent(register_path):
    assert policy_violations(load_register(register_path)) == []


def test_a_malformed_value_is_caught(register_path, tmp_path):
    with register_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rows[0]["no_lookahead"] = "MAYBE"
    rows[1]["input_decision"] = "STRICT_ISH"
    bad = tmp_path / "leakage_audit.csv"
    with bad.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=REGISTER_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    problems = policy_violations(load_register(bad))
    assert any("MAYBE" in p for p in problems)
    assert any("STRICT_ISH" in p for p in problems)
