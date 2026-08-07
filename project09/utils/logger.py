"""
utils/logger.py — Lightweight passive episode logger (per-step CSV, Project08
port). All values stored as-is; the logger never computes or aggregates.
"""

import csv
import os


class EpisodeLogger:
    """Buffers per-step data for one episode and writes to CSV on save()."""

    FIELDS = [
        "step", "U", "U_norm", "F1", "F2", "undetermined", "tracked_targets",
        "mean_bound", "well_localized_frac", "intrinsic_reward",
        "coverage_reward", "total_reward",
    ]

    def __init__(self, method, run_label, log_dir="results/logs"):
        self.method = method
        self.run_label = run_label
        safe = method.lower().replace(" ", "_").replace("@", "_")
        self.out_dir = os.path.join(log_dir, safe)
        os.makedirs(self.out_dir, exist_ok=True)
        self.filepath = os.path.join(self.out_dir, f"run_{run_label}.csv")
        self.buffer = []

    def log_step(self, step, U=None, U_norm=None, F1=None, F2=None,
                 undetermined=None, tracked_targets=None, mean_bound=None,
                 well_localized_frac=None, intrinsic_reward=None,
                 coverage_reward=None, total_reward=None):
        def _r(v, nd=4):
            if v is None:
                return None
            return round(float(v), nd)

        self.buffer.append({
            "step": step,
            "U": _r(U),
            "U_norm": _r(U_norm, 6),
            "F1": F1,
            "F2": F2,
            "undetermined": undetermined,
            "tracked_targets": tracked_targets,
            "mean_bound": _r(mean_bound),
            "well_localized_frac": _r(well_localized_frac),
            "intrinsic_reward": _r(intrinsic_reward),
            "coverage_reward": _r(coverage_reward),
            "total_reward": _r(total_reward),
        })

    def save(self):
        with open(self.filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            writer.writeheader()
            writer.writerows(self.buffer)
        self.buffer = []
