#!/usr/bin/env python3
"""Plot saved small-budget C solutions against an independent BVP reference.

No optimizer or coefficient fit is run. Endpoints are selected by the saved
1536-node pointwise energy among all ten starts. The reference is evaluated
with the original independent boundary-value solver, after endpoint selection.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

from figure_style import (ROOT, BLUE, RED, BLACK, GREEN, PALETTE, apply_style,
                          panel_geometry, save, square_panels)
apply_style(8.0)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = ROOT / 'work/pstudy-forcing-2026-11-12'
sys.path.insert(0, str(STUDY))
import pstudy as p

STEM = 'figure_forcing_solutions'
CASES = ('single_source', 'sech_dipole')
BUDGETS = (2, 4, 6)
COLORS = {2: BLUE, 4: GREEN, 6: RED}
STYLES = {2: '--', 4: '-.', 6: '-'}
AUDIT = HERE.parent / 'forcing-endpoint-audit/summary.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_saved(record, x):
    """Evaluate the saved primitive expansion using extended-precision sums."""
    blocks = copy.deepcopy(record['final_blocks'])
    for block in blocks:
        for key in ('c', 'e', 's', 'p'):
            if key in block:
                block[key] = np.array(block[key])
    centers, directions, orders, weights = p.cs.primitives(blocks, 1, 1.)
    centers = centers.numpy()[:, 0].astype(np.longdouble)
    directions = directions.numpy()[:, 0].astype(np.longdouble)
    weights = weights.numpy().astype(np.longdouble)
    coefficients = np.asarray(record['final_coefficients'], dtype=np.longdouble)
    primitive_weights = weights @ coefficients
    y = np.asarray(x, dtype=np.longdouble)[:, None] - centers[None, :]
    gaussian = np.exp(-y*y/2)
    hermite = [np.ones_like(y), y]
    for order in range(2, int(max(orders))+2):
        hermite.append(y*hermite[-1]-(order-1)*hermite[-2])
    value = np.zeros(len(x), dtype=np.longdouble)
    derivative = np.zeros_like(value)
    for order in np.unique(orders):
        use = orders == order
        w = primitive_weights[use] * directions[use]**int(order)
        value += ((-1)**int(order) * gaussian[:, use] * hermite[order][:, use]) @ w
        derivative += ((-1)**int(order+1) * gaussian[:, use] * hermite[order+1][:, use]) @ w
    return np.asarray(value, float), np.asarray(derivative, float)


def text_audit(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    intersections = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if not box.width or not box.height:
            continue
        for i, ax in enumerate(axes):
            rect = ax.get_window_extent(renderer)
            w = min(box.x1, rect.x1)-max(box.x0, rect.x0)
            h = min(box.y1, rect.y1)-max(box.y0, rect.y0)
            if w > .1 and h > .1:
                intersections.append(dict(text=text.get_text(), panel=i, pixels=[w, h]))
    assert not intersections, intersections
    return dict(text_intersections_with_data_rectangles=intersections,
                data_panels=panel_geometry(axes))


def main():
    audit = json.loads(AUDIT.read_text())['summary']
    inputs = [AUDIT, STUDY/'pstudy.py', HERE/'figure_style.py', Path(__file__).resolve()]
    src = ROOT/'numerical-section-revision/phase1-2026-11-11/src'
    inputs += [src/name for name in ('chain_solver.py', 'chain_kernels.py', 'quadrature.py', 'fit.py')]
    x = np.linspace(-4, 4, 1201)
    xq, wq = p.gl(1536)
    curves, selected = {}, {}
    for case in CASES:
        curves[case] = {}
        selected[case] = {}
        # Endpoint selection uses only saved validation energies, never the reference.
        for budget in BUDGETS:
            files = [STUDY/f'results/runs/{case}_P{budget}_s{seed}_C.json'
                     for seed in range(300, 310)]
            inputs.extend(files)
            records = [json.loads(file.read_text()) for file in files]
            record = min((r for r in records if np.isfinite(r['energy_1536'])),
                         key=lambda r: (r['energy_1536'], r['seed']))
            assert record['seed'] == audit[case][str(budget)]['C']['energy_selected']['seed']
            curves[case][budget] = evaluate_saved(record, x)[0]
            uq, uxq = p.reference_fields(case, xq)
            vq, vxq = evaluate_saved(record, xq)
            checked_error = float(np.sqrt(wq@((vq-uq)**2+(vxq-uxq)**2) /
                                          (wq@(uq*uq+uxq*uxq))))
            assert abs(checked_error-record['relative_H1_error']) < 5e-5
            selected[case][budget] = dict(seed=record['seed'],
                energy_1536=record['energy_1536'],
                relative_H1_error_recorded=record['relative_H1_error'],
                relative_H1_error_rechecked=checked_error,
                source=f'work/pstudy-forcing-2026-11-12/results/runs/{case}_P{budget}_s{record["seed"]}_C.json')
        curves[case]['reference'] = p.reference_fields(case, x)[0]

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 3.05))
    fig.subplots_adjust(left=.11, right=.985, bottom=.17, top=.79, wspace=.38)
    square_panels(axes)
    for ax, case, title in zip(axes, CASES,
                              ('(a) Single-source forcing', r'(b) Asymmetric sech$^2$ forcing')):
        ax.axhline(0, color=BLACK, lw=.45, alpha=.22, zorder=0)
        ax.plot(x, curves[case]['reference'], color=BLACK, lw=1.65, zorder=5)
        for budget in BUDGETS:
            ax.plot(x, curves[case][budget], color=COLORS[budget], ls=STYLES[budget],
                    lw=1.25, zorder=budget)
        ax.set(xlim=(-4, 4), ylim=(-2.4, 2.4), xticks=[-4, -2, 0, 2, 4],
               yticks=[-2, -1, 0, 1, 2], xlabel=r'$x$', ylabel=r'Solution $u(x)$')
        ax.set_title(title, fontsize=8.2, pad=9)
        ax.tick_params(labelsize=7.4, pad=3)
        ax.grid(axis='y')
        for line in ax.lines:
            if line.get_transform() == ax.transData:
                y = line.get_ydata()
                assert np.min(y) >= -2.4 and np.max(y) <= 2.4
    handles = [Line2D([], [], color=BLACK, lw=1.65, label='Reference')]
    handles += [Line2D([], [], color=COLORS[budget], ls=STYLES[budget], lw=1.25,
                       label=rf'C, $P={budget}$') for budget in BUDGETS]
    fig.legend(handles=handles, ncol=4, loc='upper center', bbox_to_anchor=(.55, .99),
               fontsize=7.8, handlelength=2.35, columnspacing=1.3)
    layout = text_audit(fig, axes)
    save(fig, HERE/STEM)
    shutil.copy2(HERE/f'{STEM}.pdf', ROOT/f'paper_1111/{STEM}.pdf')
    plt.close(fig)
    sampled = {'x': x}
    for case in CASES:
        for key, values in curves[case].items():
            sampled[f'{case}_{key}'] = values
    np.savez_compressed(HERE/f'{STEM}_samples.npz', **sampled)
    provenance = dict(
        description='Saved energy-selected C endpoints at P=2,4,6; no optimization or coefficient refits.',
        selection='Minimum saved finite 1536-node pointwise energy over all ten starts; ties use seed.',
        reference='Original independent scipy.solve_bvp solve on [-14,14], tolerance 3e-11; used only for evaluation.',
        evaluation='Stored primitive expansion, accumulated in numpy.longdouble; same original primitive quadrature.',
        displayed_domain=[-4., 4.], common_solution_limits=[-2.4, 2.4],
        relative_H1_error_domain=[-12., 12.], palette=list(PALETTE),
        selected=selected, layout_audit=layout,
        inputs={str(path.relative_to(ROOT)): digest(path) for path in inputs})
    (HERE/f'{STEM}_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(dict(figure=STEM, selected=selected, layout=layout), indent=2))


if __name__ == '__main__':
    main()
