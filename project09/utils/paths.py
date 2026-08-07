"""utils/paths.py — Canonical path builders (single source of truth).

The timestamped run directory layout is the heart of the Resume-Anywhere
protocol:
  results/<experiment>/{timestamp}/
      config.json     (full config + git hash + git status, written BEFORE run)
      run.log         (tee'd stdout/stderr of the run)
      DONE            (marker written ONLY after full completion)
      <artifacts>     (CSVs, .pt checkpoints, figures)

Scripts must NOT take an output path argument — they build it from the
experiment name + current timestamp. Never delete result dirs; rename to
`*_DISABLED` if they must be retired.
"""

import os
import re
import time

from project09.config import RESULTS_DIR


def sanitize_name(name):
    return re.sub(r"[^\w\-]", "_", name)


def timestamp_str(t):
    """Format a time.struct_time (or datetime-like via time.strftime)."""
    return time.strftime("%Y%m%d_%H%M%S", t)


def run_dir(experiment, t):
    """results/<experiment>/{timestamp} — created by the caller."""
    safe = sanitize_name(experiment)
    return os.path.join(RESULTS_DIR, safe, timestamp_str(t))


def raw_csv_path(out_dir, method, run=None):
    """Per-method raw CSV inside a timestamped run dir.

    Campaign runs (multiple seeds) share one per-method CSV, appended
    incrementally (resumable). Trace runs write per-run files with run index.
    """
    safe = sanitize_name(method)
    if run is None:
        return os.path.join(out_dir, f"raw__{safe}.csv")
    return os.path.join(out_dir, f"raw__{safe}__run{run}.csv")


def checkpoint_path(out_dir, tag):
    """latest.pt / best.pt / ckpt_<step>.pt inside a run dir."""
    return os.path.join(out_dir, f"{tag}.pt")
