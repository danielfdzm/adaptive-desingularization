#!/usr/bin/env python3
"""Restyle the saved local-pair tests; never rerun or alter the experiments."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
from figure_style import (ROOT, BLUE, RED, BLACK, GREEN, PALETTE, apply_style,
                          panel_geometry, save, square_panels)

HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'work/sinum-phase3-2026-11-11/predictions'
spec = importlib.util.spec_from_file_location('saved_local_results', SOURCE / 'figure_local.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
apply_style(7.7)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, LogLocator, MultipleLocator, NullFormatter, NullLocator


def audit_text(fig, axes):
    """Require every visible text extent to be outside all plotting rectangles."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems = []
    checked = 0
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if box.width == 0 or box.height == 0:
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
                continue  # Reference rules may use a blended axes/data transform.
            for x, y in line.get_xydata():
                if not (xmin <= x <= xmax and ymin <= y <= ymax):
                    raise RuntimeError(f'Clipped data point {(x, y)} in {ax.get_title()}')
                data_points += 1
    return dict(checked_text_extents=checked, intersections_with_data_rectangles=problems,
                checked_data_vertices=data_points, clipped_data_vertices=0,
                panel_geometry=panel_geometry(axes))


def decades(axis, values):
    axis.set_major_locator(FixedLocator(values))
    axis.set_minor_locator(LogLocator(base=10, subs=(1.,), numticks=40))
    axis.set_minor_formatter(NullFormatter())


def main():
    gtols, medians, seeds, toy_slopes = source.toy_sweep()
    pb, pb_slopes = source.pb_sweep()
    records = source.trajectories()
    colors = dict(NC=GREEN, T=RED, C=BLACK)
    styles = dict(NC='-.', T='--', C=':')
    fig, axes = plt.subplots(1, 3, figsize=(5.1, 2.15))
    fig.subplots_adjust(left=.105, right=.985, bottom=.22, top=.70, wspace=.73)
    square_panels(axes)

    ax = axes[0]
    for method in ('NC', 'T', 'C'):
        ax.plot(gtols, medians[method], color=colors[method], ls=styles[method],
                lw=1.05, marker='o', ms=2.5, mec=colors[method], mfc=colors[method])
        rows = pb[method]
        ax.plot([r['gtol'] for r in rows], [r['error'] for r in rows],
                color=colors[method], ls='none', marker='+' if method == 'C' else 's',
                ms=4 if method == 'C' else 3.2, mfc='white', mew=.65, zorder=6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(.3, 2e-15)
    ax.set_ylim(1e-17, 3)
    decades(ax.xaxis, [1e-1, 1e-5, 1e-9, 1e-13])
    decades(ax.yaxis, [1e-1, 1e-5, 1e-9, 1e-13, 1e-17])
    ax.set_xlabel(r'Gradient tolerance $g$', labelpad=5)
    ax.set_ylabel(r'Relative $H^1$ error', labelpad=3)
    ax.set_title('(a) Attainable accuracy', fontsize=7.7)

    selected = [r for r in records if r['final']['relative_error'] < .5]
    for ax, method, quantity in ((axes[1], 'NC', 'h'), (axes[2], 'T', 't')):
        for variance, color in ((1., BLACK), (.99, BLUE)):
            for row in selected:
                if row['method'] != method or row['variance'] != variance:
                    continue
                trace = row['trace']
                values = [point[quantity] for point in trace]
                ax.plot([point['iteration'] for point in trace], values,
                        color=color, ls=styles[method], lw=.7)
                if method == 'T':
                    first_zero = source.t_zero_iteration(trace)
                    if first_zero is not None:
                        ax.plot(first_zero, 0, marker='*', ls='none', color=color,
                                mec=color, mew=.4, ms=5.4 if variance == 1 else 3.8,
                                zorder=5 if variance == 1 else 6)
        ax.set_xlabel('L-BFGS-B iteration', labelpad=5)
        ax.set_xlim(left=-1)
    axes[1].set_yscale('log')
    axes[1].axhline(1e-12, color=BLACK, lw=.6, ls=':')
    axes[1].set_ylim(3e-13, 3)
    decades(axes[1].yaxis, [1, 1e-4, 1e-8, 1e-12])
    axes[1].xaxis.set_major_locator(MultipleLocator(25))
    axes[1].set_ylabel(r'Half-gap $h_k$ (NC)', labelpad=3)
    axes[1].set_title('(b) Pair collapse', fontsize=7.7)

    # A true zero is represented on a symlog axis, never replaced by a positive value.
    axes[2].set_yscale('symlog', linthresh=1e-28, linscale=.65)
    axes[2].set_ylim(-2e-28, 30)
    axes[2].set_yticks([0, 1e-20, 1e-10, 1])
    axes[2].set_yticklabels(['$0$', '$10^{-20}$', '$10^{-10}$', '$10^{0}$'])
    axes[2].yaxis.set_minor_locator(NullLocator())
    axes[2].xaxis.set_major_locator(MultipleLocator(5))
    axes[2].set_ylabel(r'Squared half-gap $t_k$ (C)', labelpad=3)
    axes[2].set_title(r'(c) Reaching $t=0$', fontsize=7.7)

    for ax in axes:
        ax.grid(axis='y', which='major')
        ax.tick_params(axis='both', which='major', labelsize=6.8, pad=2)
        ax.tick_params(which='minor', length=1.3)

    method_handles = [Line2D([], [], color=colors[m], ls=styles[m], lw=1.2,
                            label=label) for m, label in (
        ('NC', 'NC: no completion'), ('T', 'C: completion'), ('C', r'C$_{\rm d}$: discrete'))]
    slope_handles = [
        Line2D([], [], color=BLACK, lw=1.1, label=r'$\kappa=0$'),
        Line2D([], [], color=BLUE, lw=1.1, label=r'$\kappa>0$'),
        Line2D([], [], color=BLACK, ls='none', marker='*', ms=5, label=r'Permanent $t=0$'),
    ]
    fig.legend(handles=method_handles, ncol=3, loc='upper center', bbox_to_anchor=(.52, .985),
               fontsize=7.1, handlelength=2.1, columnspacing=1.25)
    fig.legend(handles=slope_handles, ncol=3, loc='upper center', bbox_to_anchor=(.52, .865),
               fontsize=7.1, handlelength=2.1, columnspacing=2.9)
    layout = audit_text(fig, axes)
    save(fig, HERE / 'figure_local')
    shutil.copy2(HERE / 'figure_local.pdf', ROOT / 'paper_1111/figure_local.pdf')
    plt.close(fig)

    inputs = [SOURCE / 'figure_local.py', SOURCE / 'results/vp_gtol_sweep.json',
              SOURCE / 'results/vp_trajectories.json'] + sorted((SOURCE / 'results/pb_sweep').glob('*.json'))
    provenance = dict(
        source='Saved records only; no optimization or refitting.',
        palette=list(PALETTE), successful_pair_seeds=seeds,
        toy_slopes=toy_slopes, pb_slopes=pb_slopes,
        trajectory_counts={f'{m}_s{s}': sum(r['method'] == m and r['variance'] == s for r in selected)
                           for m in ('NC', 'T') for s in (1., .99)},
        panel_c_axis='Symlog, linear threshold 1e-28; stars mark the first permanent exact zero.',
        caption_requirements='Circles: successful-start pair medians; open squares: PB start; pluses: PB C_d. Dotted floor in (b): h=1e-12.',
        layout_audit=layout,
        inputs={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (HERE / 'figure_local_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(dict(figure='local', layout=layout, trajectory_counts=provenance['trajectory_counts'])))


if __name__ == '__main__':
    main()
