#!/usr/bin/env python3
"""Re-evaluate the ten frozen single-source endpoints on two reference domains.

No optimization is rerun: C/NC/SC states are read from the original records.
Outputs retain individual errors, reference residuals, quadrature sensitivity,
source hashes, and package versions. The larger-domain solve is not training data.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import csv
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'work/independent-tests-2026-11-08/forcing_pair'
spec = importlib.util.spec_from_file_location('frozen_forcing', SOURCE / 'run.py')
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def relative_error(x, u, du, row):
    B, Bx, *_ = original.basis(x, row['beta'], row['h']**2)
    c = np.asarray(row['coefficients'])
    v, dv = c @ B, c @ Bx
    w = original.weights(x)
    return float(np.sqrt(w @ ((v-u)**2+(dv-du)**2) / (w @ (u*u+du*du))))


def main():
    protocol = json.loads((SOURCE / 'holdout_protocol.json').read_text())
    records = json.loads((SOURCE / 'holdout/results.json').read_text())
    lookup = {(r['seed'], r['method']): r for r in records}
    results, references = [], []
    for case in protocol['cases']:
        seed = case['seed']
        refs = {L: original.reference(case['a'], case['A'], case['lam'],
                                      L, 1e-10) for L in (22, 24)}
        values = {}
        for L, ref in refs.items():
            x = np.linspace(-L, L, 800*L+1)  # dx=.0025, as in archived validation
            u, du = ref.sol(x)
            fine_x = np.linspace(-L, L, 1600*L+1)
            fine_u, fine_du = ref.sol(fine_x)
            values[L] = x, u, du, fine_x, fine_u, fine_du
            np.savez_compressed(HERE / f'reference-{seed}-L{L}.npz',
                                x=x, u=u, du=du, bvp_mesh=ref.x,
                                bvp_state=ref.y)
        xx, u22, du22, *_ = values[22]
        u24i, du24i = refs[24].sol(xx)
        x24, u24, du24, *_ = values[24]
        norm24 = original.weights(x24) @ (u24*u24+du24*du24)
        ref_change = float(np.sqrt(original.weights(xx) @
                         ((u22-u24i)**2+(du22-du24i)**2) / norm24))
        references.append(dict(**case, domain22_vs24_relative_h1=ref_change,
            max_bvp_rms_residual={str(L):float(np.max(r.rms_residuals))
                                  for L,r in refs.items()},
            bvp_nodes={str(L):len(r.x) for L,r in refs.items()}))
        for method in ('C', 'NC', 'SC'):
            row = lookup[seed, method]
            error = {L:relative_error(*values[L][:3], row) for L in refs}
            fine = {L:relative_error(*values[L][3:], row) for L in refs}
            results.append(dict(**case, method=method,
                error_domain22=error[22], error_domain24=error[24],
                domain_error_change=abs(error[24]-error[22]),
                error_domain22_refined=fine[22], error_domain24_refined=fine[24],
                max_quadrature_error_change=max(abs(fine[L]-error[L]) for L in refs),
                archived_error_domain24=row['relative_h1'],
                archived_error_difference=abs(error[24]-row['relative_h1']),
                beta=row['beta'], h=row['h'], coefficients=row['coefficients']))
        print(seed, 'reference change', f'{ref_change:.4e}',
              'error change', f"{results[-1]['domain_error_change']:.4e}", flush=True)
        (HERE / 'per-case-results.json').write_text(json.dumps(results, indent=2)+'\n')
        (HERE / 'reference-checks.json').write_text(json.dumps(references, indent=2)+'\n')
    pairs = []
    for seed in range(900,910):
        pair = {r['method']:r for r in results if r['seed']==seed}
        pairs.append(dict(seed=seed, C=pair['C']['error_domain24'],
                          NC=pair['NC']['error_domain24'],
                          SC=pair['SC']['error_domain24'],
                          absolute_C_NC_difference=abs(pair['C']['error_domain24']-
                                                       pair['NC']['error_domain24'])))
    summary = dict(n_cases=10, methods={m:dict(
        median_relative_h1_domain24=float(np.median([r['error_domain24'] for r in results if r['method']==m])),
        median_relative_h1_domain22=float(np.median([r['error_domain22'] for r in results if r['method']==m])))
        for m in ('C','NC','SC')},
        max_paired_absolute_C_NC_error_difference=max(r['absolute_C_NC_difference'] for r in pairs),
        max_domain_error_change=max(r['domain_error_change'] for r in results),
        max_domain_error_change_C_NC=max(r['domain_error_change'] for r in results if r['method'] in ('C','NC')),
        max_reference_domain_change=max(r['domain22_vs24_relative_h1'] for r in references),
        max_quadrature_error_change=max(r['max_quadrature_error_change'] for r in results),
        max_archived_error_difference=max(r['archived_error_difference'] for r in results),
        reference_tolerance=1e-10, quadrature_step=.0025,
        quadrature_refinement_step=.00125)
    (HERE / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (HERE / 'paired-errors.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(pairs[0]))
        writer.writeheader();writer.writerows(pairs)
    with (HERE / 'per-case-results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader();writer.writerows(results)
    inputs = [Path(__file__), SOURCE/'run.py', SOURCE/'protocol.json',
              SOURCE/'holdout_protocol.json', SOURCE/'holdout/results.json',
              SOURCE/'holdout/references.json']
    (HERE/'provenance.json').write_text(json.dumps(dict(
        python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
        platform=platform.platform(),
        command='python3 numerical-section-revision/sinum-revision-2026-11-10/forcing/revalidate.py',
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    ),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
