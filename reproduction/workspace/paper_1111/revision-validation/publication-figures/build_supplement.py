#!/usr/bin/env python3
"""Restyle the eight current supplement figures from their frozen source records.

The phase-1 builder owns cohort selection and numerical evaluation. This wrapper
changes only the presentation, captures its figures before saving, and exports
them here. No optimizer is invoked and no source record is modified.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import figure_style as style
from figure_style import BLUE, RED, BLACK, GREEN
style.apply_style(8)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from matplotlib.text import Text
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = style.ROOT
PAPER = ROOT / 'paper_1111'
PHASE = ROOT / 'numerical-section-revision/phase1-2026-11-11/figures/build_figures.py'
STATS = ROOT / 'work/sinum-phase3-2026-11-11/stats'
INPUTS, OUTPUTS, AUDITS = {}, {}, {}
COLORS = {'S': BLUE, 'NC': GREEN, 'C': RED, 'Cd': BLACK, 'NCa': GREEN}
STYLES = {'S': '-', 'NC': '-.', 'C': '--', 'Cd': ':', 'NCa': (0, (1, 1))}
LABELS = {'S': 'Ordinary (S)', 'NC': 'No completion (NC)', 'C': 'Completion (C)',
          'Cd': r'Discrete completion ($C_{\rm d}$)', 'NCa': 'NC, averaged evaluator'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def method_handles(methods):
    return [Line2D([], [], color=COLORS[m], ls=STYLES[m], lw=1.25, label=LABELS[m]) for m in methods]


def marker_handle(marker, label, color=BLACK):
    return Line2D([], [], color=color, marker=marker, ls='none', ms=4,
                  mfc='white' if marker in ('o', '>') else color, label=label)


def legend(fig, handles, ncol, y=.015):
    for old in list(fig.legends):
        old.remove()
    return fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.55, y),
                      ncol=ncol, fontsize=7.6, columnspacing=1.0, handlelength=1.8,
                      handletextpad=.5, labelspacing=.55)


def normalize(fig):
    """Use the fixed hues and neutral black for framing, grids and all text."""
    palette = [to_rgba(c)[:3] for c in (BLUE, RED, BLACK, GREEN)]
    def neutral(c):
        try:
            rgba = to_rgba(c)
        except (ValueError, TypeError):
            return c
        if rgba[3] == 0 or rgba[:3] == (1., 1., 1.):
            return c
        if rgba[:3] in palette:
            return c
        return BLACK
    for ax in fig.axes:
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_color(BLACK)
        for line in ax.lines:
            line.set_color(neutral(line.get_color()))
            line.set_markeredgecolor(neutral(line.get_markeredgecolor()))
            line.set_markerfacecolor(neutral(line.get_markerfacecolor()))
            line.set_alpha(1)
            line.set_clip_on(True)
        for collection in ax.collections:
            for prop in ('facecolors', 'edgecolors'):
                vals = getattr(collection, 'get_' + prop)()
                if len(vals):
                    vals = [to_rgba(neutral(v)) for v in vals]
                    getattr(collection, 'set_' + prop)(vals)
        for grid in ax.get_xgridlines() + ax.get_ygridlines():
            grid.set_color(BLACK); grid.set_alpha(.10); grid.set_linewidth(.45)
        ax.tick_params(which='both', colors=BLACK, labelsize=7.4, pad=3)
        ax.title.set_fontsize(8.5)
        ax.xaxis.label.set_fontsize(8)
        ax.yaxis.label.set_fontsize(8)
    for text in fig.findobj(Text):
        text.set_color(BLACK)
        text.set_fontsize(min(text.get_fontsize(), 8.5))


def audit(fig, data_axes, stem):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overlaps = []
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        # Hidden ticks on shared axes have invisible parents, but their Text
        # objects can still report visible; inspect their explicit visibility.
        box = text.get_window_extent(renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        for i, ax in enumerate(data_axes):
            ab = ax.get_window_extent(renderer)
            w = min(box.x1, ab.x1) - max(box.x0, ab.x0)
            h = min(box.y1, ab.y1) - max(box.y0, ab.y0)
            if w > .1 and h > .1:
                overlaps.append({'text': text.get_text(), 'panel': i, 'overlap_pixels': [float(w), float(h)]})
    geometry = style.panel_geometry(data_axes)
    assert all(abs(panel['height_over_width'] - 1) < 1e-10
               for panel in geometry), (stem, geometry)
    AUDITS[stem] = {'text_intersections_with_data_rectangles': overlaps,
                    'data_panels': len(data_axes), 'panel_geometry': geometry,
                    'palette': list(style.PALETTE)}
    assert not overlaps, (stem, overlaps)


def export(fig, stem, data_axes):
    for ax in data_axes:
        # A square plotting rectangle is independent of the data-unit aspect.
        # Preserve limits for images and zooms as well as for line plots.
        ax.set_aspect('auto')
    style.square_panels(data_axes)
    normalize(fig)
    audit(fig, data_axes, stem)
    style.save(fig, HERE / stem)
    OUTPUTS[stem] = {ext: digest(HERE / f'{stem}.{ext}') for ext in ('pdf', 'png', 'svg')}
    shutil.copy2(HERE / f'{stem}.pdf', PAPER / f'{stem}.pdf')
    plt.close(fig)
    return OUTPUTS[stem]


def format_traces(fig, stem):
    axes = fig.axes
    for text in list(fig.texts):
        text.remove()
    n = len(axes)
    fig.set_size_inches(5.35, 3.25 if n == 2 else 2.65)
    width, height = fig.get_size_inches()
    if n == 2:
        positions = [(.125, .30, .355, .355*width/height),
                     (.635, .30, .355, .355*width/height)]
    else:
        positions = [(.115 + j*.305, .30, .265, .265*width/height) for j in range(3)]
    for ax, pos in zip(axes, positions):
        ax.set_position(pos)
        ax.set_title(ax.get_title(), y=1.01, pad=8)
        ax.set_xlabel('Coefficient fits' if ax.get_xscale() != 'log' else 'Coefficient fits (log)')
    if stem == 'figure_variance_family':
        for ax in axes:
            ax.set_xlabel('Optimizer iterations $+1$')
        handles = method_handles(('S', 'NC', 'C')) + [marker_handle('*', r'Pair reaches $t=0$', RED)]
        handles += [Line2D([], [], color=BLACK, ls=':', label='Completed reference')]
        legend(fig, handles, 3)
    elif stem == 'figure_order':
        handles = method_handles(('S', 'NC', 'C'))
        handles += [marker_handle('o', 'Block completion'), marker_handle('>', 'Fit limit'),
                    marker_handle('x', r'Return with $E>10^{-6}$')]
        legend(fig, handles, 3)
    elif stem == 'figure_recovery':
        handles = method_handles(('NC', 'C'))
        handles += [marker_handle('.', 'Endpoint (80 digits)'), marker_handle('o', 'Block completion'),
                    marker_handle('*', r'Pair reaches $t=0$'), marker_handle('x', 'Abnormal return')]
        legend(fig, handles, 3)
    else:
        legend(fig, method_handles(('S', 'C')) + [marker_handle('>', 'Fit limit')], 3)
    return export(fig, stem, axes)


def format_solution1d(fig, stem):
    axes = fig.axes
    fig.set_size_inches(5.35, 5.85)
    width, height = fig.get_size_inches()
    for artist in list(fig.artists):
        if isinstance(artist, ConnectionPatch):
            artist.remove()
    for patch in list(axes[0].patches):
        patch.remove()
    for line in list(axes[0].lines):
        if line.get_transform() != axes[0].transData:
            line.remove()
    for text in list(fig.texts):
        text.remove()
    positions = [(.125, .565, .35, .35*width/height),
                 (.64, .565, .35, .35*width/height)]
    positions += [(.125+j*.31, .145, .245, .245*width/height) for j in range(3)]
    for j, (ax, pos) in enumerate(zip(axes, positions)):
        ax.set_position(pos)
        extra = [t.get_text() for t in ax.texts]
        for text in list(ax.texts):
            text.remove()
        title = ax.get_title()
        if extra:
            title += '\n' + ' '.join(extra)
        ax.set_title(title, y=1.025, pad=5, va='bottom')
        for spine in ax.spines.values():
            spine.set_visible(spine.spine_type in ('left', 'bottom'))
    # The zooms are independent panels: removing crossing connector lines
    # leaves their coordinates and scientific interpretation unchanged.
    legend(fig, [Line2D([], [], color=BLACK, label='Target')]+method_handles(('S', 'C')), 3, y=.95)
    fig.text(.55, .038, r'S: Gaussian weights. C: coefficients of $G$, $G\prime$, $G\prime\prime$.',
             ha='center', va='bottom', fontsize=7.5)
    fig.text(.55, .008, r'Square: singleton; diamond: pair with $t>0$; star: completed block.',
             ha='center', va='bottom', fontsize=7.5)
    return export(fig, stem, axes)


def format_solutionpb(fig, stem):
    axes, bars = fig.axes[:6], fig.axes[6:]
    fig.set_size_inches(5.35, 5.7)
    width, height = fig.get_size_inches()
    for j, ax in enumerate(axes):
        ax.set_position((.095+(j%3)*.305, .69 if j < 3 else .28,
                         .235, .235*width/height))
        ax.set_title(ax.get_title(), y=1.015, pad=7)
        for text in list(ax.texts):
            fig.text(.2125+(j%3)*.305, .657, text.get_text(), ha='center', va='bottom', fontsize=7.6)
            text.remove()
    signed = LinearSegmentedColormap.from_list('paper_signed', [BLUE, 'white', RED])
    error = LinearSegmentedColormap.from_list('paper_error', ['white', BLUE])
    for ax in axes[:3]:
        ax.images[0].set_cmap(signed)
    for ax in axes[3:5]:
        ax.images[0].set_cmap(error)
    # Initial centers use green; reference centers use the common black.
    axes[5].collections[0].set_color(GREEN)
    axes[5].yaxis.tick_right()
    axes[5].yaxis.set_label_position('right')
    for bar, bottom in zip(bars, (.602, .178)):
        bar.set_position((.095, bottom, .845, .017))
        bar.xaxis.labelpad = 3
        bar.tick_params(labelsize=7.2, pad=2)
    handles = [marker_handle('x', 'Target center'), marker_handle('o', 'Ordinary (S)', BLUE),
               marker_handle('*', r'Pair, $t=0$', RED), marker_handle('D', r'Pair, $t>0$', RED),
               marker_handle('^', 'Singleton', RED), marker_handle('.', 'Initial center', GREEN)]
    legend(fig, handles, 3, y=.015)
    return export(fig, stem, axes)


def phase_save(fig, stem, title):
    if stem == 'figure_solution_1d':
        return format_solution1d(fig, stem)
    if stem == 'figure_solution_pb':
        return format_solutionpb(fig, stem)
    return format_traces(fig, stem)


def csv_rows(path):
    INPUTS[str(path.relative_to(ROOT))] = digest(path)
    with path.open() as f:
        return list(csv.DictReader(f))


def ecdf_figure(panels, stem):
    style.apply_style(8)
    fig, axes = plt.subplots(1, 3, figsize=(5.35, 2.8))
    width, height = fig.get_size_inches()
    present = set()
    for i, (ax, title, rows, methods, wide) in enumerate(zip(axes, *zip(*panels))):
        ax.set_position((.12+i*.305, .34, .26, .26*width/height))
        for threshold in (1e-6, 1e-10):
            ax.axvline(threshold, color=BLACK, ls=':', lw=.55)
        for m in methods:
            values = np.sort(np.maximum([float(r['error']) for r in rows if r['method'] == m], 1e-16))
            assert len(values)
            y = np.arange(1, len(values)+1)/len(values)
            ax.step(np.r_[1e-16, values, 10], np.r_[0, y, 1], where='post',
                    color=COLORS[m], ls=STYLES[m], lw=1.15, label=LABELS[m])
            present.add(m)
        ax.set_xscale('log')
        ax.set_xlim((1e-16, 2) if wide else (1e-9, 1e-2))
        ax.set_xticks([1e-16, 1e-8, 1] if wide else [1e-8, 1e-5, 1e-2])
        ax.set_ylim(0, 1.02); ax.set_yticks([0, .25, .5, .75, 1])
        ax.grid(axis='y')
        ax.set_title(f'({"abc"[i]}) {title}', pad=9)
        ax.set_xlabel(r'Final $H^1$ error $E$')
        if i:
            ax.tick_params(labelleft=False)
    axes[0].set_ylabel('Fraction of starts\n' + r'with error $\leq E$')
    legend(fig, method_handles([m for m in ('S', 'NC', 'Cd', 'C', 'NCa') if m in present]), 2 if len(present)==4 else 3, y=.01)
    return export(fig, stem, axes)


def main():
    phase = module(PHASE, 'current_supplement_source')
    phase.plot_style = SimpleNamespace(BLUE=BLUE, RED=RED, BLACK=BLACK, GREEN=GREEN,
                                     apply_style=lambda *a, **kw: style.apply_style(8))
    phase.COLORS = {m: COLORS[m] for m in ('S', 'NC', 'C')}
    phase.save = phase_save
    for fn in (phase.figure_order, phase.figure_recovery, phase.figure_dimension,
               phase.figure_variance, phase.figure_solution1d, phase.figure_solutionpb):
        style.apply_style(8)
        fn()
    INPUTS.update(phase.INPUTS)
    one = csv_rows(STATS / 'cohort1d-per-run.csv')
    pb = [r for r in csv_rows(STATS / 'cohort-pb-per-run.csv') if r['method'] != 'Cda']
    methods1 = tuple(m for m in ('S', 'NC', 'Cd', 'C') if any(r['method']==m for r in one))
    methods_pb = ('S', 'NC', 'Cd', 'C')
    ecdf_figure([(r'Independent, $r=2$', [r for r in one if r['order']=='2'], methods1, True),
                 (r'Independent, $r=3$', [r for r in one if r['order']=='3'], methods1, True),
                 (r'PB, $d=5,10,20$', pb, methods_pb, False)], 'figure_ecdf')
    ecdf_figure([(rf'PB, $d={d}$', [r for r in pb if r['dimension']==str(d)], methods_pb+('NCa',), False)
                 for d in (5, 10, 20)], 'figure_ecdf_pb_dimensions')
    for path in (PHASE, HERE/'figure_style.py', Path(__file__)):
        INPUTS[str(path.relative_to(ROOT))] = digest(path)
    assert all(digest(ROOT/path)==sha for path, sha in INPUTS.items())
    (HERE/'supplement_provenance.json').write_text(json.dumps({
        'inputs': INPUTS, 'outputs': OUTPUTS, 'layout_audits': AUDITS,
        'cohort_facts': phase.FACTS,
        'notes': ['Plotting only; phase-1 cohort selection and numerical evaluation retained.',
                  'Every data panel has a measured 1:1 box aspect; colorbars retain their separate external layout.',
                  'All text and legends lie outside data rectangles; connector lines removed from 1D zooms.',
                  'NCa uses green dotted lines; signed fields use blue-white-red and error fields white-blue.']}, indent=2)+'\n')
    print(json.dumps({'outputs': list(OUTPUTS), 'input_files': len(INPUTS), 'audits': AUDITS}, indent=2))


if __name__ == '__main__':
    main()
