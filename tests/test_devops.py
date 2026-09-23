"""
tests/test_devops.py — Resume-Anywhere protocol gate: golden trio, DONE
marker, [SKIP] on rerun, config.json with git hash written BEFORE the run.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.config import DONE_MARKER
from src.devops.runner import ExperimentRunner


BASE_CONFIG = {
    "experiment": "devops_test",
    "phase": 0,
    "regime": "MATE-4v2-9-v0",
    "seed": 0,
    "num_runs": 1,
    "reward_lambda": 0.5,
    "chao_variant": "bias_cap",
    "max_steps": 100,
}


def test_runner_writes_golden_trio(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.paths.RESULTS_DIR", str(tmp_path))
    runner = ExperimentRunner(experiment="golden", phase=1,
                              timestamp="20260101_000000", config=BASE_CONFIG)
    assert runner.begin() is True
    # config.json exists BEFORE any computation
    assert os.path.exists(runner.config_path)
    with open(runner.config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    for k in ("experiment", "phase", "regime", "git_commit",
              "git_status", "git_dirty"):
        assert k in cfg, f"config.json missing {k}"
    # run.log exists
    assert os.path.exists(runner.log_path)
    # no DONE yet
    assert not os.path.exists(runner.done_path)
    runner.finish()
    assert os.path.exists(runner.done_path)
    assert open(runner.done_path, encoding="utf-8").read().strip()


def test_rerun_skips_when_done(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.paths.RESULTS_DIR", str(tmp_path))
    runner = ExperimentRunner(experiment="golden", phase=1,
                              timestamp="20260101_000000", config=BASE_CONFIG)
    runner.begin()
    runner.finish()
    runner2 = ExperimentRunner(experiment="golden", phase=1,
                               timestamp="20260101_000000", config=BASE_CONFIG)
    assert runner2.skip_if_done() is True
    assert runner2.begin() is False


def test_runner_rejects_missing_config_keys(tmp_path, monkeypatch):
    monkeypatch.setattr("src.utils.paths.RESULTS_DIR", str(tmp_path))
    runner = ExperimentRunner(experiment="bad", phase=0,
                              timestamp="20260101_000000",
                              config={"experiment": "bad"})
    with pytest.raises(ValueError):
        runner.begin()


def test_done_marker_only_on_success(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("src.utils.paths.RESULTS_DIR", str(tmp_path))
    from src.devops.runner import RunContext
    ctx = RunContext(experiment="ctx", phase=0, timestamp="20260101_000000",
                     config=BASE_CONFIG)
    try:
        with ctx as c:
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not os.path.exists(ctx.runner.done_path)
    out = capsys.readouterr().out
    assert "[SKIP]" not in out
