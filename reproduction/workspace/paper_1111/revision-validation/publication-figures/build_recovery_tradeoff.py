"""Paired recovery accuracy/work distributions from the frozen per-run records."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

import figure_style as style
style.apply_style(8.0)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from build_algorithm import layout_check

HERE = Path(__file__).resolve().parent
INPUT = style.ROOT / 'work/sinum-phase3-2026-11-11/stats'


def records(name):
    with (INPUT / name).open() as f:
        return list(csv.DictReader(f))


def paired(rows, key):
    a = {key(r): r for r in rows if r['method'] == 'C'}
    b = {key(r): r for r in rows if r['method'] == 'NC'}
    assert a.keys() == b.keys(), 'The paired comparison must retain every start.'
    data = []
    for k in sorted(a):
        ea, eb = float(a[k]['error']), float(b[k]['error'])
        assert ea > 0 and eb > 0
        data.append(dict(key=k, error_ratio=ea / eb,
                         fit_ratio=float(a[k]['fits']) / float(b[k]['fits'])))
    return data


def main():
    one, pb = records('cohort1d-per-run.csv'), records('cohort-pb-per-run.csv')
    groups = [
        ('Order 2 (100 starts)', style.BLUE, '-', paired(
            [r for r in one if r['order'] == '2'], lambda r: (r['order'], r['seed']))),
        ('Order 3 (100 starts)', style.RED, '--', paired(
            [r for r in one if r['order'] == '3'], lambda r: (r['order'], r['seed']))),
        ('PB (150 starts)', style.GREEN, '-.', paired(
            pb, lambda r: (r['dimension'], r['geometry'], r['seed']))),
    ]
    assert [len(g[3]) for g in groups] == [100, 100, 150]
    fig, axes = plt.subplots(1, 2, figsize=(5.35, 3.0))
    fig.subplots_adjust(left=.115, right=.985, bottom=.18, top=.79, wspace=.32)
    style.square_panels(axes)
    ranges = [(1e-16, 1e8), (.1, 4.5)]
    for ax, metric, lim in zip(axes, ['error_ratio', 'fit_ratio'], ranges):
        for label, color, ls, data in groups:
            x = np.sort([r[metric] for r in data])
            assert x.min() >= lim[0] and x.max() <= lim[1]
            y = np.arange(1, len(x) + 1) / len(x)
            ax.step(np.r_[lim[0], x, lim[1]], np.r_[0., y, 1.], where='post',
                    color=color, linestyle=ls, linewidth=1.35)
        ax.axvline(1, color=style.BLACK, linestyle=':', linewidth=.85)
        ax.set_xscale('log')
        ax.set_xlim(*lim)
        ax.set_ylim(-.02, 1.035)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.grid(axis='y')
        ax.minorticks_off()
    axes[0].set_title('(a) Accuracy change')
    axes[0].set_xlabel(r'Relative error ratio $E_{\rm C}/E_{\rm NC}$')
    axes[0].set_ylabel('Fraction of paired starts')
    axes[0].set_xticks([1e-16, 1e-10, 1e-4, 1, 1e8])
    axes[1].set_title('(b) Work change')
    axes[1].set_xlabel('Coefficient-fit ratio C / NC')
    axes[1].set_xticks([.1, .25, .5, 1, 2, 4])
    axes[1].set_xticklabels(['0.1', '0.25', '0.5', '1', '2', '4'])
    handles = [Line2D([], [], color=c, linestyle=ls, linewidth=1.4, label=label)
               for label, c, ls, _ in groups]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.53, .99),
               ncol=3, fontsize=7.1, columnspacing=1.1, handlelength=2.1)
    layout = layout_check(fig, axes)
    style.save(fig, HERE / 'figure_recovery_tradeoff')
    plt.close(fig)
    metadata = dict(
        inputs={str((INPUT / f).relative_to(style.ROOT)):
                hashlib.sha256((INPUT / f).read_bytes()).hexdigest()
                for f in ['cohort1d-per-run.csv', 'cohort-pb-per-run.csv']},
        interpretation='All paired starts, including failures. Ratios below one favor C. Fits are not wall time.',
        layout_check=layout,
        groups={label: dict(n=len(data), rows=data,
                           lower_error=sum(r['error_ratio'] < 1 for r in data),
                           fewer_fits=sum(r['fit_ratio'] < 1 for r in data))
                for label, _, _, data in groups})
    (HERE / 'recovery_tradeoff_provenance.json').write_text(json.dumps(metadata, indent=2) + '\n')
    shutil.copy2(HERE / 'figure_recovery_tradeoff.pdf', HERE.parents[1] / 'figure_recovery_tradeoff.pdf')


if __name__ == '__main__':
    main()
