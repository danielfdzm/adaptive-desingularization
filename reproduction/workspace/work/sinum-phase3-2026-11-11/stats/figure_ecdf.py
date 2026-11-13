#!/usr/bin/env python3
"""Empirical distribution functions of the final relative H1 error (Phase-3 cohorts).

Main figure (figure_ecdf): (a), (b) one-dimensional independent recovery, orders 2 and 3, 100 starts each;
(c) first-order Poisson--Boltzmann recovery pooled over d = 5, 10, 20 (150 starts).
Supplement figure (figure_ecdf_pb_dimensions): the Poisson--Boltzmann cohort for each dimension (50 starts).
Palette and labels follow adaptive-desingularization/experiments/plot_style.py; C is the paper's method
(pairs in t = h^2), C_d the discrete-completion ablation.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'adaptive-desingularization' / 'experiments'))
import plot_style as ps            # noqa: E402
ps.apply_style(9.0)
import matplotlib.pyplot as plt     # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np                  # noqa: E402

COLORS = dict(S=ps.BLUE, NC=ps.GREEN, C=ps.RED, Cd=ps.BLACK, NCa=ps.GRAY)   # paper palette
STYLES = dict(S='-', NC='-.', C='--', Cd=':', NCa='-.')
LABELS = dict(S='Ordinary (S)', NC='No completion (NC)', C='Completion (C)', Cd=r'Discrete completion (C$_{\rm d}$)',
              NCa='NC, averaged pair evaluator')
ORDER = ('S', 'NC', 'Cd', 'C', 'NCa')
FLOOR = 1e-16


def read(path):
    with path.open() as fh:
        return list(csv.DictReader(fh))


def ecdf(ax, errors, m):
    e = np.sort(np.maximum(np.asarray(errors, float), FLOOR))
    y = np.arange(1, len(e) + 1) / len(e)
    ax.step(np.r_[FLOOR, e, 10.], np.r_[0., y, 1.], where='post', color=COLORS[m], ls=STYLES[m],
            lw=1.25 if m != 'NCa' else .9, zorder=1 if m == 'NCa' else 2)


WIDE = dict(xlim=(FLOOR, 2.), xticks=(1e-16, 1e-12, 1e-8, 1e-4, 1e0))
NARROW = dict(xlim=(1e-9, 1e-2), xticks=(1e-8, 1e-6, 1e-4, 1e-2))


def panel(ax, letter, title, rows, methods, axis=WIDE):
    for t in (1e-6, 1e-10):
        ax.axvline(t, color=ps.LIGHT_GRAY, lw=.7, zorder=0)
    for m in methods:
        ecdf(ax, [float(r['error']) for r in rows if r['method'] == m], m)
    ps.panel_title(ax, letter, title)
    ax.set_xscale('log')
    ax.set_xlim(*axis['xlim'])
    ax.set_ylim(0, 1.02)
    ax.set_xticks(list(axis['xticks']))
    ax.grid(axis='y', alpha=.25, lw=.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel(r'Final relative $H^1$ error $E$')


def figure(panels, stem):
    fig, axes = plt.subplots(1, len(panels), figsize=(ps.FULL_WIDTH, 2.45), gridspec_kw=dict(wspace=.32))
    for ax, args in zip(axes, panels):
        panel(ax, *args)
    axes[0].set_ylabel('Fraction of starts\nwith error $\\leq E$')
    present = [m for m in ORDER if any(m in p[3] for p in panels)]
    handles = [Line2D([], [], color=COLORS[m], ls=STYLES[m], lw=1.25 if m != 'NCa' else .9, label=LABELS[m]) for m in present]
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), bbox_to_anchor=(.5, 1.1))
    ps.save_figure(fig, HERE / stem)
    plt.close(fig)


def main():
    one = read(HERE / 'cohort1d-per-run.csv')
    pb = read(HERE / 'cohort-pb-per-run.csv')
    # First-order PB: NC and C_d use the exact pair evaluator of C (amendment 3); the averaged-evaluator NC
    # (NCa) appears only in the supplement figure.
    pb = [r for r in pb if r['method'] != 'Cda']
    m1 = tuple(m for m in ORDER if any(r['method'] == m for r in one))
    mb = tuple(m for m in ORDER if any(r['method'] == m for r in pb))
    mb_main = tuple(m for m in mb if m not in ('NCa', 'Cda'))
    figure([('a', 'Independent, $r=2$', [r for r in one if r['order'] == '2'], m1),
            ('b', 'Independent, $r=3$', [r for r in one if r['order'] == '3'], m1),
            ('c', 'PB, $r=1$, $d=5,10,20$', pb, mb_main, NARROW)], 'figure_ecdf')
    figure([('a', 'PB, $r=1$, $d=5$', [r for r in pb if r['dimension'] == '5'], mb, NARROW),
            ('b', 'PB, $r=1$, $d=10$', [r for r in pb if r['dimension'] == '10'], mb, NARROW),
            ('c', 'PB, $r=1$, $d=20$', [r for r in pb if r['dimension'] == '20'], mb, NARROW)], 'figure_ecdf_pb_dimensions')
    counts = {f'{lab} {m}': sum(r['method'] == m for r in rows)
              for lab, rows in (('1d r=2', [r for r in one if r['order'] == '2']), ('1d r=3', [r for r in one if r['order'] == '3']),
                                ('pb', pb)) for m in ORDER}
    meta = dict(inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (HERE / 'cohort1d-per-run.csv', HERE / 'cohort-pb-per-run.csv')},
                methods_1d=m1, methods_pb=mb, runs_per_curve={k: v for k, v in counts.items() if v},
                note='errors below 1e-16 are drawn at 1e-16; vertical lines mark the thresholds 1e-6 and 1e-10')
    (HERE / 'figure_ecdf_provenance.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    main()
