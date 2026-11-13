#!/usr/bin/env python3
"""Figure: numerical tests of Proposition 3.1 / Corollary 3.2 (local analysis of a pair collision).

(a) Final relative error against the gradient tolerance (ftol = 0): one-dimensional zero-gap toy
    (u* = G', sigma = 1, P = 2, refitted coefficients; median over the starts that reach the target)
    and the first-order PB start (d, geometry, seed) = (5, 0, 10) in the production solver.
(b) Half-gap h_k of NC against L-BFGS-B iteration, kappa = 0 (s = 1) and kappa > 0 (s = 0.99).
(c) Squared half-gap t_k of the t-chart, kappa = 0 and kappa > 0; stars: the iteration from which t_k = 0.
Writes figure_local.{pdf,svg,png} and figure_local_provenance.json (all plotted numbers come from results/).

Style (11 Nov 2026): the conventions of the other row plots of the paper, built by
numerical-section-revision/phase1-2026-11-11/figures/build_figures.py (figure_recovery, figure_order):
6.2 in wide with the same margins in inches, 10 pt labels and titles, 8.8 pt tick labels, light y grid,
no top/right spines, method names and line styles of plot_style, marker types in dark gray in the legend
above the panels, stars where a pair reaches t = 0. The previous version is in
work/archive-before-figure1-rowstyle-2026-11-11/.
"""
import collections
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'adaptive-desingularization' / 'experiments'))
import plot_style as ps            # noqa: E402
ps.apply_style(10)                 # as build_figures.py
import matplotlib.pyplot as plt     # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, LogLocator, MultipleLocator, NullFormatter, NullLocator  # noqa: E402
import numpy as np                  # noqa: E402

# Record keys: T is the paper's C (pairs in t = h^2), red dashed; C is the paper's C_d, black dotted.
COL = dict(NC=ps.GREEN, T=ps.RED, C=ps.BLACK)     # paper palette: blue, red, green, black (no purple/orange)
STY = dict(NC='-.', T='--', C=':')
LAB = dict(NC='No completion (NC)', T='Completion (C)', C=r'Discrete completion (C$_{\rm d}$)')
KCOL = {1.0: ps.BLACK, .99: ps.BLUE}               # kappa = 0 (s = 1) and kappa > 0 (s = 0.99)
MARK = '#333333'                                   # marker-type legend entries, as in figure_recovery
RES = HERE / 'results'
ZERO = 1e-30        # where t = 0 is drawn in panel (c)
# Row-plot geometry: Fig. 3 (figure_recovery) is 6.2 x 2.65 in with bottom 0.58 in and top 0.64 in margins;
# this figure keeps those margins, with 0.1 in more axes height for its longer y labels.
WIDTH, HEIGHT = 6.2, 2.75
LEFT, RIGHT, BOTTOM, TOP, WSPACE = .105, .965, .212, .769, .46


def decorate(ax):
    """Axis styling of the row plots (build_figures.decorate)."""
    ax.grid(axis='y', which='major', color='#dddddd', linewidth=.45)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(which='major', labelsize=8.8)
    ax.tick_params(which='minor', length=1.3)


def log_decades(axis, majors):
    axis.set_major_locator(FixedLocator(majors))
    axis.set_minor_locator(LogLocator(base=10, subs=(1.0,), numticks=40))
    axis.set_minor_formatter(NullFormatter())


def t_zero_iteration(trace):
    """First iteration from which t_k = 0 up to the end of the trace; None if the run ends with t > 0."""
    k = None
    for p in reversed(trace):
        if p['t'] != 0.:
            break
        k = p['iteration']
    return k


def save(fig, stem):
    """Export as build_figures.save (tight box, 0.035 in pad, no time stamps); SVG kept as before."""
    for ext in ('pdf', 'svg', 'png'):
        meta = {'Creator': 'figure_local.py', 'Title': 'Tests of the local analysis of a pair collision'}
        if ext == 'pdf':
            meta.update(CreationDate=None, ModDate=None)
        elif ext == 'svg':
            meta = {'Date': None}
        fig.savefig(stem.with_suffix('.' + ext), dpi=300 if ext == 'png' else 600, bbox_inches='tight',
                    pad_inches=.035, metadata=meta)


def toy_sweep():
    S = json.loads((RES / 'vp_gtol_sweep.json').read_text())
    by = collections.defaultdict(list)
    for r in S:
        by[(r['method'], r['seed'])].append(r)
    good = sorted({seed for (m, seed), rs in by.items() if min(r['final']['relative_error'] for r in rs) < 1e-2
                   and all(min(x['final']['relative_error'] for x in by[(mm, seed)]) < 1e-2 for mm in ('NC', 'C', 'T'))})
    gtols = sorted({r['gtol'] for r in S}, reverse=True)
    med = {}
    for m in ('NC', 'C', 'T'):
        med[m] = [float(np.median([next(r['final']['relative_error'] for r in by[(m, s)] if r['gtol'] == g) for s in good]))
                  for g in gtols]
    slopes = {}
    for m in ('NC', 'C', 'T'):
        vals = []
        for s in good:
            rs = sorted(by[(m, s)], key=lambda r: -r['gtol'])
            emin = min(r['final']['relative_error'] for r in rs)
            sel = [r for r in rs if 'PGTOL' in (r['last_return_message'] or '') and r['final']['relative_error'] > 10 * emin]
            if m == 'C':
                sel = [r for r in sel if r['completions'] > 0]
            if len(sel) >= 3:
                vals.append(float(np.polyfit(np.log10([r['gtol'] for r in sel]), np.log10([r['final']['relative_error'] for r in sel]), 1)[0]))
        slopes[m] = dict(median=float(np.median(vals)), low=float(min(vals)), high=float(max(vals)), n=len(vals))
    return gtols, med, good, slopes


def pb_sweep():
    rows = collections.defaultdict(list)
    for p in sorted((RES / 'pb_sweep').glob('pb_d5_g0_s10_*_gtol*.json')):
        r = json.loads(p.read_text())
        m = r['phase3_method']
        # SciPy's own message: the frozen classifier labels a restart that starts below gtol (zero step)
        # as 'numerical_stagnation', although the projected-gradient test was met.
        last = r['optimizer_returns'][-1]['message'] if r['optimizer_returns'] else ''
        rows[m].append(dict(gtol=r['optimizer_overrides']['gtol'], error=r['final_relative_error'], fits=r['fits'],
                            stop=r['stop'], last=last, completions=r['completions']))
    for m in rows:
        rows[m].sort(key=lambda x: -x['gtol'])
    slopes = {}
    for m, rs in rows.items():
        emin = min(x['error'] for x in rs)
        sel = [x for x in rs if 'PGTOL' in (x['last'] or '') and x['error'] > 10 * emin]
        if m == 'C':
            sel = [x for x in sel if x['completions'] > 0]
        if len(sel) >= 3:
            slopes[m] = dict(slope=float(np.polyfit(np.log10([x['gtol'] for x in sel]), np.log10([x['error'] for x in sel]), 1)[0]),
                             n=len(sel), gtol_range=[min(x['gtol'] for x in sel), max(x['gtol'] for x in sel)],
                             floor=emin)
    return rows, slopes


def joint_slopes():
    """Pre-registered joint coefficient/geometry protocol (toy.py): slopes of log E against log gtol."""
    S = json.loads((RES / 'gtol_sweep_toy.json').read_text())
    by = collections.defaultdict(list)
    for r in S:
        by[(r['method'], r['seed'])].append(r)
    out = {}
    for m in ('NC', 'C', 'T'):
        vals = []
        for seed in range(10):
            rs = sorted(by[(m, seed)], key=lambda r: -r['gtol'])
            emin = min(r['relative_error'] for r in rs)
            if emin > 1e-2:
                continue
            sel = [r for r in rs if 'PGTOL' in (r['last_return_message'] or '') and r['relative_error'] > 10 * emin]
            if m == 'C':
                sel = [r for r in sel if r['completions'] > 0]
            if len(sel) >= 3:
                vals.append(float(np.polyfit(np.log10([r['gtol'] for r in sel]), np.log10([r['relative_error'] for r in sel]), 1)[0]))
        out[m] = dict(median=float(np.median(vals)), low=float(min(vals)), high=float(max(vals)), n=len(vals))
    return out


def joint_trajectories():
    """Pre-registered joint protocol: NC contraction of h_k, iterations to the bound, T identification."""
    T = json.loads((RES / 'trajectories.json').read_text())
    out = dict(contraction={}, floor_iteration=[], t_identification={})
    for s in (1.0, .99):
        out['contraction'][str(s)] = []
        out['t_identification'][str(s)] = []
    for r in T:
        if r['final']['relative_error'] >= .5:
            continue
        tr = r['trace']
        if r['method'] == 'NC':
            sel = [(p['iteration'], p['h']) for p in tr if 1e-11 < p['h'] <= 1e-2]
            if len(sel) >= 4:
                out['contraction'][str(r['variance'])].append(
                    float(10 ** np.polyfit([a for a, b in sel], np.log10([b for a, b in sel]), 1)[0]))
            if r['variance'] == .99:
                out['floor_iteration'].append(next((p['iteration'] for p in tr if p['h'] <= 1.000001e-12), None))
        if r['method'] == 'T':
            k = None
            for i in range(len(tr) - 1, -1, -1):
                if tr[i]['t'] == 0.:
                    k = tr[i]['iteration']
                else:
                    break
            out['t_identification'][str(r['variance'])].append(k)
    return out


def trajectories():
    return json.loads((RES / 'vp_trajectories.json').read_text())


def main():
    gtols, med, good, toy_slopes = toy_sweep()
    pb, pb_slopes = pb_sweep()
    traj = trajectories()
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, HEIGHT))
    fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP, wspace=WSPACE)
    ax = axes[0]
    for m in ('NC', 'T', 'C'):
        ax.plot(gtols, med[m], color=COL[m], ls=STY[m], lw=1.1, marker='o', ms=2.6, mfc=COL[m], mec=COL[m])
    for m in ('NC', 'T', 'C'):
        if m in pb:
            xs = [x['gtol'] for x in pb[m]]
            ys = [x['error'] for x in pb[m]]
            if m == 'C':       # C_d lies on C at this scale: a plus inside the open square
                ax.plot(xs, ys, color=COL[m], ls='none', marker='+', ms=4.2, mew=.8, zorder=6)
            else:
                ax.plot(xs, ys, color=COL[m], ls='none', marker='s', ms=3.6, mfc='white', mec=COL[m], mew=.7, zorder=5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # reference slopes as short segments in the empty upper right of the panel, labels inside the axes
    g = np.array([1e-5, 1e-9])
    ax.plot(g, 1.0 * (g / 1e-5) ** (2 / 3), color=ps.GRAY, lw=.8)
    g = np.array([1e-6, 1e-9])
    ax.plot(g, 1e-2 * (g / 1e-6), color=ps.GRAY, lw=.8)
    ax.text(5e-10, (1e-4) ** (2 / 3), r'slope $2/3$', color=ps.GRAY, fontsize=8.4, va='center')
    ax.text(5e-10, 1e-5, r'slope $1$', color=ps.GRAY, fontsize=8.4, va='center')
    ax.set_xlim(.3, 2e-15)
    ax.set_ylim(1e-17, 3)
    log_decades(ax.xaxis, [1e-1, 1e-5, 1e-9, 1e-13])
    log_decades(ax.yaxis, [1e-1, 1e-5, 1e-9, 1e-13, 1e-17])
    ax.set_xlabel('Gradient tolerance $g$', fontsize=10)
    ax.set_ylabel(r'Final relative $H^1$ error', fontsize=10)
    ax.set_title('(a) Error and tolerance', fontsize=10, pad=5)
    ax = axes[1]
    for s, col in KCOL.items():
        for r in traj:
            if r['variance'] == s and r['method'] == 'NC' and r['final']['relative_error'] < .5:
                it = [p['iteration'] for p in r['trace']]
                hh = [p['h'] for p in r['trace']]
                ax.plot(it, hh, color=col, ls=STY['NC'], lw=.85, alpha=.68)
    ax.set_yscale('log')
    ax.axhline(1e-12, color='.45', lw=.8, ls=':')
    ax.set_ylim(3e-13, 3)
    log_decades(ax.yaxis, [1e0, 1e-4, 1e-8, 1e-12])
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.set_xlabel('L-BFGS-B iteration', fontsize=10)
    ax.set_ylabel('Half-gap $h_k$ (NC)', fontsize=10)
    ax.text(40, 3e-4, r'$\kappa=0$', color=KCOL[1.0], fontsize=9)
    ax.text(17, 3e-9, r'$\kappa>0$', color=KCOL[.99], fontsize=9)
    ax.set_title('(b) Collapse of the pair', fontsize=10, pad=5)
    ax = axes[2]
    star = {1.0: 6.6, .99: 4.4}     # kappa = 0 larger and underneath, so a shared iteration shows both stars
    for s, col in KCOL.items():
        for r in traj:
            if r['variance'] == s and r['method'] == 'T' and r['final']['relative_error'] < .5:
                it = [p['iteration'] for p in r['trace']]
                tt = [max(p['t'], ZERO) for p in r['trace']]
                ax.plot(it, tt, color=col, ls=STY['T'], lw=.85, alpha=.68)
                k = t_zero_iteration(r['trace'])
                if k is not None:
                    ax.plot([k], [ZERO], marker='*', ms=star[s], color=col, mec='black', mew=.3, ls='none',
                            zorder=5 if s == 1.0 else 6)
    ax.set_yscale('log')
    ax.set_yticks([ZERO, 1e-20, 1e-10, 1e0])
    ax.set_yticklabels(['$0$', '$10^{-20}$', '$10^{-10}$', '$10^{0}$'])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(ZERO / 30, 30)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel('L-BFGS-B iteration', fontsize=10)
    ax.set_ylabel('Squared half-gap $t_k$ (C)', fontsize=10)
    ax.text(9.5, 1e-12, r'$\kappa=0$', color=KCOL[1.0], fontsize=9)
    ax.text(10.5, 1e-27, r'$\kappa>0$', color=KCOL[.99], fontsize=9)
    ax.set_title('(c) Identification of $t=0$', fontsize=10, pad=5)
    for a in axes:
        decorate(a)
    # legend as in figure_recovery: method lines, then marker types in dark gray; two rows above the panels
    methods = [Line2D([0], [0], color=COL[m], ls=STY[m], lw=1.3, label=LAB[m]) for m in ('NC', 'T', 'C')]
    markers = [Line2D([0], [0], color=MARK, ls='-', lw=.9, marker='o', ms=2.8, label='Pair problem, median'),
               Line2D([0], [0], color=MARK, ls='none', marker='s', ms=3.8, mfc='white', mew=.7, label='PB start'),
               Line2D([0], [0], color=MARK, ls='none', marker='+', ms=4.6, mew=.8, label=r'PB start, C$_{\rm d}$'),
               Line2D([0], [0], color=MARK, ls='none', marker='*', ms=5, label=r'Pair reaches $t=0$')]
    kw = dict(loc='upper center', fontsize=8.6, columnspacing=1.0, handlelength=1.8)
    fig.legend(handles=methods, ncol=3, bbox_to_anchor=((LEFT + RIGHT) / 2, 1 + .159 / HEIGHT), **kw)
    fig.legend(handles=markers, ncol=4, bbox_to_anchor=((LEFT + RIGHT) / 2, 1 + .004 / HEIGHT), **kw)
    save(fig, HERE / 'figure_local')
    plt.close(fig)
    ident = {s: [] for s in (1.0, .99)}
    decay = {s: [] for s in (1.0, .99)}
    floor_iter = []
    for r in traj:
        if r['final']['relative_error'] >= .5:
            continue
        if r['method'] == 'T':
            k = None
            for i in range(len(r['trace']) - 1, -1, -1):
                if r['trace'][i]['t'] == 0.:
                    k = r['trace'][i]['iteration']
                else:
                    break
            ident[r['variance']].append(k)
        if r['method'] == 'NC':
            sel = [(p['iteration'], p['h']) for p in r['trace'] if 1e-11 < p['h'] <= 1e-2]
            if len(sel) >= 4:
                decay[r['variance']].append(float(10 ** np.polyfit([a for a, b in sel], np.log10([b for a, b in sel]), 1)[0]))
            if r['variance'] == .99:
                floor_iter.append(next((p['iteration'] for p in r['trace'] if p['h'] <= 1.000001e-12), None))
    prov = dict(toy_good_seeds=good, toy_slopes=toy_slopes, pb_slopes=pb_slopes,
                joint_protocol_slopes=joint_slopes(), joint_protocol_trajectories=joint_trajectories(),
                pb_rows={m: v for m, v in pb.items()},
                t_identification_iteration={str(k): v for k, v in ident.items()},
                nc_h_contraction_per_iteration={str(k): v for k, v in decay.items()},
                nc_iteration_reaching_h_floor_kappa_positive=floor_iter,
                inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [RES / 'vp_gtol_sweep.json', RES / 'vp_trajectories.json',
                                  RES / 'gtol_sweep_toy.json', RES / 'trajectories.json']})
    (HERE / 'figure_local_provenance.json').write_text(json.dumps(prov, indent=2) + '\n')
    print(json.dumps({k: v for k, v in prov.items() if k not in ('pb_rows', 'inputs')}, indent=1))


if __name__ == '__main__':
    main()
