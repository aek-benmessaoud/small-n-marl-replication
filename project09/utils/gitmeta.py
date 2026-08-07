"""
utils/gitmeta.py — Git identity captured into config.json (protocol).

Every experiment's config.json must record the git commit hash and a
clean/dirty status snapshot so results are reproducible to a revision.
"""

import os
import subprocess

from project09.config import PROJECT_ROOT


def git_rev_parse():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or ""
    except Exception:  # noqa: BLE001 - never let gitmeta break a run
        return ""


def git_status_porcelain():
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def git_meta():
    return {
        "git_commit": git_rev_parse(),
        "git_status": git_status_porcelain(),
        "git_dirty": bool(git_status_porcelain()),
    }
