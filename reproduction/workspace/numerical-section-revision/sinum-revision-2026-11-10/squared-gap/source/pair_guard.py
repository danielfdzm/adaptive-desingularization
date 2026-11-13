"""Conservative new-direction guard; no changes to an existing pair chart."""
import math
import numpy as np


def eligible_pair(solver, distance, i, j, sigma):
    counts = getattr(solver, "guard_counts", None)
    if counts is None:
        counts = solver.guard_counts = dict(tested=0, eligible=0, unresolved=0, nonfinite=0)
    counts["tested"] += 1
    distance = float(distance)
    reason = None
    if not math.isfinite(distance):
        reason = "nonfinite"
    elif distance / sigma <= 32 * (np.finfo(np.float64).eps / 2):
        reason = "unresolved"
    if reason:
        counts[reason] += 1
        solver.events.append(dict(kind="pair_guard_rejected", reason=reason,
                                  fits=solver.fits, indices=[i, j],
                                  distance=distance if math.isfinite(distance) else None))
        return False
    counts["eligible"] += 1
    return True
