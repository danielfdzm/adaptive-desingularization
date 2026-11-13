#!/usr/bin/env python3
"""Phase 1 (11 Nov 2026): pair-evaluator control for the directional (chain) solver.

In the chain solver, all methods evaluate a pair by the eight-node Gauss--Legendre formulas in h, except that C
(method "tpair") switches to the Taylor series in t = h^2 below t = T_SWITCH sigma^2.  This control reruns NC
("stable") with the same switch: a pair with h^2 < T_SWITCH sigma^2 is evaluated by the series of C in t = h^2
(the half-gap chart and everything else are unchanged).  src/chain_solver.py is imported unchanged; only its
module-level function `primitives` is wrapped at run time.
Usage: python3.13 chain_series_runner.py --problem linear1d --seed 3 --output runs/evaluator-chain/x.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
import torch                                   # noqa: E402
import chain_solver as cs                      # noqa: E402
import chain_experiments as ce                 # noqa: E402

_original = cs.primitives
SWITCHED = {"evaluations_with_series_pair": 0}


def primitives_series(blocks, d, sigma=1.0):
    out, used = [], False
    for b in blocks:
        if b["kind"] == "pair" and "h" in b and "t" not in b:
            h = b["h"]
            hv = float(h.detach()) if torch.is_tensor(h) else float(h)
            if hv * hv < cs.T_SWITCH * sigma * sigma:
                b = {k: v for k, v in b.items() if k != "h"}
                b["t"] = h * h
                used = True
        out.append(b)
    SWITCHED["evaluations_with_series_pair"] += used
    return _original(out, d, sigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["linear1d", "second1d"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--method", default="stable", choices=["stable", "adaptive"])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    cs.primitives = primitives_series
    out = ce.run_one((args.problem, 1, 0, args.seed, args.method))
    out["phase1_control"] = "NC with the Taylor-series pair evaluator of C below t = T_SWITCH sigma^2 (h-chart unchanged)"
    out["series_switch"] = dict(SWITCHED)
    srcs = sorted((HERE / "src").glob("*.py")) + [Path(__file__)]
    out["source_sha256"] = {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in srcs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(dict(problem=args.problem, seed=args.seed, method=args.method, error=out["final_relative_error"],
                          fits=out["fits"], stop=out["stop"], series=SWITCHED["evaluations_with_series_pair"])), flush=True)


if __name__ == "__main__":
    main()
