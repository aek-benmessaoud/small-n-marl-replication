"""
devops/runner.py — Resume-Anywhere experiment runner.

PROTOCOL (locked):
  - Every experiment lives in  results/<experiment>/{timestamp}/
      config.json   written BEFORE the run (config + git hash + git status)
      run.log       tee'd stdout/stderr of the run
      DONE          marker written ONLY after full completion
      <artifacts>   CSVs / .pt / figures
  - On entry: if results/<experiment>/{timestamp}/DONE exists -> [SKIP] and
    exit 0 (never recompute). Otherwise create the dir and write config.json.
  - Never delete result dirs. Rename to `*_DISABLED` if they must be retired.
  - Per-phase `pip freeze` snapshot written alongside config.json
    (requirements_phase{phase}.txt).
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from project09.config import (
    DONE_MARKER, PROJECT_ROOT, REQUIRED_CONFIG_KEYS, RESULTS_DIR,
)
from project09.utils.gitmeta import git_meta
from project09.utils.paths import run_dir


class ExperimentRunner:
    """Manages one timestamped experiment directory + the golden trio."""

    def __init__(self, experiment, phase=None, timestamp=None, config=None):
        now = time.localtime()
        ts = (timestamp or
              time.strftime("%Y%m%d_%H%M%S", now))
        self.experiment = experiment
        self.phase = phase
        self.timestamp = ts
        self.out_dir = run_dir(experiment, time.strptime(ts, "%Y%m%d_%H%M%S"))
        self.done_path = os.path.join(self.out_dir, DONE_MARKER)
        self.config_path = os.path.join(self.out_dir, "config.json")
        self.log_path = os.path.join(self.out_dir, "run.log")
        self.config = dict(config or {})

    # ---- lifecycle ---------------------------------------------------------
    def skip_if_done(self):
        """If DONE exists, print [SKIP] and return True (caller should exit)."""
        if os.path.exists(self.done_path):
            print(f"[SKIP] {self.experiment} already complete: {self.out_dir}",
                  flush=True)
            return True
        return False

    def begin(self):
        """Create the run dir and write config.json + pip freeze BEFORE the
        actual computation. Returns True if the run may proceed."""
        if self.skip_if_done():
            return False
        os.makedirs(self.out_dir, exist_ok=True)
        self._validate_config()
        self._write_config()
        self._write_pip_freeze()
        # touch run.log so the golden trio exists before any computation
        open(self.log_path, "a", encoding="utf-8").close()
        return True

    def finish(self):
        """Write the DONE marker (full completion only)."""
        with open(self.done_path, "w", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n")

    # ---- config.json -------------------------------------------------------
    def _validate_config(self):
        missing = [k for k in REQUIRED_CONFIG_KEYS if k not in self.config]
        if missing:
            raise ValueError(
                f"config missing required keys: {missing}. Provided: "
                f"{sorted(self.config)}")

    def _write_config(self):
        payload = dict(self.config)
        payload.update({
            "experiment": self.experiment,
            "phase": self.phase,
            "timestamp": self.timestamp,
            "out_dir": self.out_dir,
            "results_dir": RESULTS_DIR,
            **git_meta(),
        })
        payload = {k: v for k, v in payload.items() if v != ""}
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
            f.write("\n")

    # ---- pip freeze --------------------------------------------------------
    def _write_pip_freeze(self):
        try:
            out = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True, text=True, timeout=120,
            ).stdout
        except Exception:  # noqa: BLE001
            out = "pip freeze unavailable"
        name = (f"requirements_phase{self.phase}.txt"
                if self.phase is not None else "requirements.txt")
        with open(os.path.join(self.out_dir, name), "w", encoding="utf-8") as f:
            f.write(out)


class Tee:
    """Duplicates writes to both the given file and stdout."""

    def __init__(self, path):
        self.path = path
        self.file = open(path, "a", encoding="utf-8", buffering=1)
        self.lock = threading.Lock()

    def write(self, data):
        with self.lock:
            self.file.write(data)
            sys.__stdout__.write(data)

    def flush(self):
        with self.lock:
            self.file.flush()
            sys.__stdout__.flush()

    def close(self):
        self.file.close()


class RunContext:
    """Context manager: creates the run dir, tees run.log, writes DONE on
    successful exit, and ensures no DONE is written on failure.

    Usage:
        with RunContext(experiment="smoke", phase=0, config={...}) as ctx:
            ... run ...
        # DONE written only if the with-block completed without exception
    """

    def __init__(self, experiment, phase=None, config=None, timestamp=None):
        self.runner = ExperimentRunner(
            experiment=experiment, phase=phase, config=config,
            timestamp=timestamp)
        self.tee = None
        self._log_backup = None

    def __enter__(self):
        if not self.runner.begin():
            sys.exit(f"[SKIP] {self.runner.experiment} already complete")
        self.tee = Tee(self.runner.log_path)
        self._log_backup = sys.stdout
        sys.stdout = self.tee
        sys.stderr = self.tee
        print(f"[RUN] {self.runner.experiment} | phase {self.runner.phase} | "
              f"out={self.runner.out_dir}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.tee is not None:
            sys.stdout = self._log_backup
            sys.stderr = self._log_backup
            self.tee.close()
        if exc_type is None:
            self.runner.finish()
            print(f"[DONE] {self.runner.experiment} | "
                  f"out={self.runner.out_dir}", flush=True)
        return False
