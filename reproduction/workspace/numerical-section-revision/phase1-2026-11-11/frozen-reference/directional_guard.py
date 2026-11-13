"""Equation (C.4): validate charts before taking any node quotient.

All public arrays use physical nodes/offsets and dimensionless stored p-hats.
Unit roundoff is eps/2. Intentional Hermite repetitions in the base are allowed;
every joined node must be resolved from every other node.
"""
from __future__ import annotations

import numpy as np

UNIT_ROUNDOFF = np.finfo(np.float64).eps / 2


class InadmissibleChart(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def node_products(nodes, base_count, sigma):
    nodes = np.asarray(nodes, dtype=float)
    if not np.isfinite(sigma) or sigma <= 0 or not np.all(np.isfinite(nodes)):
        raise InadmissibleChart("nonfinite_nodes_or_scale")
    x = nodes / sigma
    if not np.all(np.isfinite(x)):
        raise InadmissibleChart("nonfinite_dimensionless_nodes")
    threshold = 32 * UNIT_ROUNDOFF * max(1., float(np.max(np.abs(x))))
    differences = []
    for i in range(base_count, len(x)):
        diff = x[i] - np.delete(x, i)
        if np.any(np.abs(diff) <= threshold):
            raise InadmissibleChart("unresolved_joined_separation")
        differences.append(diff)
    products = np.array([np.prod(diff) for diff in differences])
    if not np.all(np.isfinite(products)) or np.any(np.abs(products) <= np.finfo(float).tiny):
        raise InadmissibleChart("unresolved_or_nonfinite_product")
    cap = min(1., float(np.ptp(x)))
    return products, cap


def guard_offsets(nodes, offsets, base_count, sigma):
    """Check physical offsets by multiplication/comparison; no node division."""
    products, cap = node_products(nodes, base_count, sigma)
    y = np.asarray(offsets, dtype=float) / sigma
    if y.ndim != 2 or len(y) != len(products) or not np.all(np.isfinite(y)):
        raise InadmissibleChart("nonfinite_or_invalid_offsets")
    if np.any(np.linalg.norm(y, axis=1) > cap * np.abs(products)):
        raise InadmissibleChart("transverse_cap")
    return products, cap


def guard_stored(nodes, p_hats, base_count, sigma):
    """Validate an optimizer trial directly in dimensionless stored coordinates."""
    products, cap = node_products(nodes, base_count, sigma)
    p_hats = np.asarray(p_hats, dtype=float)
    if p_hats.ndim != 2 or len(p_hats) != len(products) or not np.all(np.isfinite(p_hats)):
        raise InadmissibleChart("nonfinite_or_invalid_stored_coordinates")
    if np.any(np.linalg.norm(p_hats, axis=1) > cap):
        raise InadmissibleChart("stored_transverse_cap")
    return products, cap
