#!/usr/bin/env python3
"""Render the corrected saved-endpoint budget comparison, without numerical reruns."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from figure_style import (ROOT, BLUE, RED, BLACK, GREEN, PALETTE, apply_style,
                          panel_geometry, save, square_panels)
apply_style(7.7)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
import numpy as np

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent / 'forcing-endpoint-audit/summary.json'
BUDGETS = (2, 4, 6, 8, 12, 16)
CASES = ('single_source', 'sech_dipole')


def audit_text(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    checked = 0
    problems = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if not box.width or not box.height:
            continue
        checked += 1
        for i, ax in enumerate(axes):
            rect = ax.get_window_extent(renderer)
            width = min(box.x1, rect.x1) - max(box.x0, rect.x0)
            height = min(box.y1, rect.y1) - max(box.y0, rect.y0)
            if width > .5 and height > .5:
                problems.append(dict(text=text.get_text(), panel=i, width=width, height=height))
    if problems:
        raise RuntimeError(f'Text intersects data rectangles: {problems}')
    data_points = 0
    for ax in axes:
        xmin, xmax = sorted(ax.get_xlim())
        ymin, ymax = sorted(ax.get_ylim())
        for line in ax.lines:
            if line.get_transform() != ax.transData:
                continue
            for x, y in line.get_xydata():
                if not (xmin <= x <= xmax and ymin <= y <= ymax):
                    raise RuntimeError(f'Clipped data point {(x, y)} in {ax.get_title()}')
                data_points += 1
    return dict(checked_text_extents=checked, intersections_with_data_rectangles=problems,
                checked_data_vertices=data_points, clipped_data_vertices=0,
                panel_geometry=panel_geometry(axes))


def main():
    summary = json.loads(AUDIT.read_text())['summary']
    base_paths = {case: ROOT / f'work/pstudy-forcing-2026-11-12/results/baselines/{case}.json' for case in CASES}
    baselines = {case: json.loads(p.read_text()) for case, p in base_paths.items()}
    fig, axes = plt.subplots(1, 3, figsize=(5.1, 2.35))
    fig.subplots_adjust(left=.105, right=.985, bottom=.25, top=.69, wspace=.72)
    square_panels(axes)
    for ax in axes:
        ax.axvspan(7, 18, color=BLACK, alpha=.055, lw=0, zorder=0)
        ax.axvline(7, color=BLACK, alpha=.35, ls=':', lw=.55, zorder=1)

    plotted = {}
    for ax, case, title in zip(axes[:2], CASES, ('(a) Single source', r'(b) sech$^2$ forcing')):
        rows = [summary[case][str(P)]['C'] for P in BUDGETS]
        selected = [r['energy_selected']['relative_H1_error'] for r in rows]
        medians = [r['median_retained_error'] for r in rows]
        gal = [next(g for g in baselines[case]['gaussian_galerkin'] if g['P'] == P)['relative_H1_error'] for P in BUDGETS]
        fem = [r for r in baselines[case]['fem_p1'] if 2 <= r['dof']/2 <= 16]
        ax.plot(BUDGETS, selected, color=RED, ls='--', marker='^', ms=3.7, lw=1.15)
        ax.plot(BUDGETS, medians, color=RED, ls='none', marker='^', ms=3.7, mfc='white', mew=.8)
        ax.plot(BUDGETS, gal, color=BLUE, ls=':', marker='x', ms=3.5, mew=.85, lw=1.05)
        ax.plot([r['dof']/2 for r in fem], [r['relative_H1_error'] for r in fem],
                color=GREEN, ls='-', marker='.', ms=3, lw=1.)
        ax.set_yscale('log')
        ticks = [.15, .2, .3, .4, .6, .9] if case == 'single_source' else [.02, .05, .1, .2, .5]
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.yaxis.set_major_formatter(FixedFormatter([f'{v:g}' for v in ticks]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylim((.14, 1.05) if case == 'single_source' else (.016, .65))
        ax.set_ylabel(r'Relative $H^1$ error', labelpad=4)
        ax.set_title(title, fontsize=7.7)
        plotted[case] = dict(budgets=list(BUDGETS), energy_selected=selected,
                            retained_median=medians, fixed_centers=gal,
                            p1_dof=[r['dof'] for r in fem], p1_error=[r['relative_H1_error'] for r in fem],
                            retained=[r['retained'] for r in rows],
                            collided=[r['collided_retained'] for r in rows])
    for case, color, marker in (('single_source', BLUE, 'o'), ('sech_dipole', BLACK, 's')):
        data = plotted[case]
        fraction = [a/b for a, b in zip(data['collided'], data['retained'])]
        axes[2].plot(BUDGETS, fraction, color=color, marker=marker, ms=3.4, lw=1.1)
        data['completed_fraction'] = fraction
    axes[2].set_ylim(-.05, 1.08)
    axes[2].set_yticks([0, .25, .5, .75, 1])
    axes[2].set_yticklabels(['0', '.25', '.50', '.75', '1'])
    axes[2].set_ylabel('Fraction of retained runs', labelpad=4)
    axes[2].set_title('(c) Completed limits in C', fontsize=7.7)
    for ax in axes:
        ax.set_xscale('log', base=2)
        ax.set_xlim(1.8, 17.6)
        ax.set_xticks(BUDGETS)
        ax.set_xticklabels([str(P) for P in BUDGETS])
        ax.set_xlabel(r'Gaussian budget $P$', labelpad=5)
        ax.tick_params(which='major', labelsize=6.8, pad=2)
        ax.grid(axis='y', which='major')
    # Matplotlib fills columns first: column 1 C statistics; column 2 baselines;
    # column 3 the two forcing cases in (c). Every legend lies outside the panels.
    handles = [
        Line2D([], [], color=RED, ls='--', marker='^', ms=3.7, lw=1.1, label='C, energy-selected'),
        Line2D([], [], color=RED, ls='none', marker='^', ms=3.7, mfc='white', mew=.8, label='C, retained median'),
        Line2D([], [], color=BLUE, ls=':', marker='x', ms=3.5, lw=1, label='Fixed centers'),
        Line2D([], [], color=GREEN, ls='-', marker='.', ms=3, lw=1, label=r'P1, $2P$ unknowns'),
        Line2D([], [], color=BLUE, ls='-', marker='o', ms=3.4, lw=1, label='Single source (c)'),
        Line2D([], [], color=BLACK, ls='-', marker='s', ms=3.4, lw=1, label=r'sech$^2$ forcing (c)'),
    ]
    fig.legend(handles=handles, ncol=3, loc='upper center', bbox_to_anchor=(.52, .985),
               fontsize=7., handlelength=2.0, columnspacing=1.5, labelspacing=.75)
    fig.text(.545, .045, r'Shaded region: exploratory endpoints, $P\geq 8$', ha='center', fontsize=7.1)
    layout = audit_text(fig, axes)
    save(fig, HERE / 'figure_budget')
    shutil.copy2(HERE / 'figure_budget.pdf', ROOT / 'paper_1111/figure_budget.pdf')
    plt.close(fig)
    inputs = [AUDIT] + list(base_paths.values())
    provenance = dict(source='Corrected saved endpoint audit; no optimization or coefficient refits.',
                      selection='Minimum finite 1536-node pointwise energy across all ten starts, without reference-error filtering.',
                      exploratory='All P>=8 endpoints; retained fractions use the original breakdown convention.',
                      palette=list(PALETTE), plotted=plotted, layout_audit=layout,
                      inputs={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (HERE / 'figure_budget_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(dict(figure='budget', layout=layout,
                         retained={case: data['retained'] for case, data in plotted.items()})))


if __name__ == '__main__':
    main()
