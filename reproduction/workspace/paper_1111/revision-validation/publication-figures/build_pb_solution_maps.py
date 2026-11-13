#!/usr/bin/env python3
"""PB solution slices and matched spatial errors, evaluated from saved states.

No fitting or optimization is performed. The example is the first seed of
geometry 0 in dimension 10, also used for the main algorithm illustration.
The NC and C files are those used in the current recovery cohort table.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import figure_style as style
style.apply_style(8)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.ticker import NullLocator
import mpmath as mp
import numpy as np

from build_algorithm import layout_check

HERE = Path(__file__).resolve().parent
ROOT = style.ROOT
SOURCE = ROOT / 'work/sinum-phase3-2026-11-11/cohort_pb'
STEM = 'figure_pb_solution_maps'
SIGMA = .45
LD = np.longdouble
PATHS = {
    'NC': SOURCE / 'results_nc_exact/pb_d10_g0_s10_NCx.json',
    'C': SOURCE / 'results/pb_d10_g0_s10_T.json',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def terms(blocks, coefficients, points):
    """Cancellation-free exact pair columns, including their t=0 limit."""
    points = np.asarray(points, dtype=LD)
    variance = LD(SIGMA)**2
    columns = []
    k = 0
    for block in blocks:
        offset = points - np.asarray(block['c'], dtype=LD)
        radius2 = np.sum(offset * offset, axis=1)
        if block['kind'] == 'single':
            columns.append(LD(coefficients[k]) * np.exp(-radius2 / (2 * variance)))
            k += 1
            continue
        assert block['kind'] == 'pair'
        direction = np.asarray(block['e'], dtype=LD)
        axial = np.sum(offset * direction, axis=1) / variance
        t = LD(0) if block.get('completed', False) else LD(
            block.get('t', block.get('h', 0.)**2))
        factor = np.exp(-(radius2 + t) / (2 * variance))
        y = axial * np.sqrt(t)
        small = np.abs(y) < LD(1e-4)
        safe_y = np.where(small, LD(1), y)
        sinhc = np.where(small, 1 + y*y/6 + y**4/120,
                         np.sinh(safe_y) / safe_y)
        columns.extend((LD(coefficients[k]) * factor * np.cosh(y),
                        LD(coefficients[k+1]) * factor * axial * sinhc))
        k += 2
    assert k == len(coefficients)
    return np.asarray(columns)


def compensated_sum(columns):
    total = np.zeros(columns.shape[1], dtype=LD)
    correction = np.zeros_like(total)
    for column in columns:
        updated = total + column
        correction += np.where(abs(total) >= abs(column),
                               (total - updated) + column,
                               (column - updated) + total)
        total = updated
    return total + correction


def high_precision_value(blocks, coefficients, point):
    variance = mp.mpf(SIGMA)**2
    values = []
    k = 0
    for block in blocks:
        offset = [mp.mpf(float(x)) - mp.mpf(float(c))
                  for x, c in zip(point, block['c'])]
        radius2 = mp.fsum(x*x for x in offset)
        if block['kind'] == 'single':
            values.append(mp.mpf(coefficients[k]) * mp.exp(-radius2/(2*variance)))
            k += 1
            continue
        axial = mp.fsum(x * mp.mpf(e) for x, e in zip(offset, block['e'])) / variance
        t = mp.mpf(0) if block.get('completed', False) else mp.mpf(
            block.get('t', block.get('h', 0.)**2))
        factor = mp.exp(-(radius2 + t)/(2*variance))
        y = axial * mp.sqrt(t)
        values.extend((mp.mpf(coefficients[k]) * factor * mp.cosh(y),
                       mp.mpf(coefficients[k+1]) * factor * axial *
                       (mp.sinh(y)/y if y else mp.mpf(1))))
        k += 2
    return mp.fsum(values)


def main():
    runs = {name: json.loads(path.read_text()) for name, path in PATHS.items()}
    for field in ('target_blocks', 'target_coefficients', 'initial_centers'):
        assert runs['NC'][field] == runs['C'][field]
    target = runs['C']
    blocks, coefficients = target['target_blocks'], target['target_coefficients']
    centers = np.asarray([block['c'] for block in blocks])
    origin = centers.mean(axis=0)
    _, singular_values, basis = np.linalg.svd(centers-origin, full_matrices=False)
    for vector in basis:
        if vector[np.argmax(abs(vector))] < 0:
            vector *= -1
    grid = np.linspace(-1.3, 1.3, 881)
    s1, s2 = np.meshgrid(grid, grid, indexing='xy')
    planes = []
    worst = 0.
    for second in (1, 2):
        points = (origin + s1.ravel()[:, None] * basis[0]
                  + s2.ravel()[:, None] * basis[second])
        target_terms = terms(blocks, coefficients, points)
        fields = {'target': compensated_sum(target_terms)}
        errors = {}
        for name, run in runs.items():
            columns = terms(run['blocks'], run['coefficients'], points)
            fields[name] = compensated_sum(columns)
            errors[name] = abs(compensated_sum(np.concatenate((columns, -target_terms))))
        indices = set(np.linspace(0, len(points)-1, 21, dtype=int))
        for values in errors.values():
            indices.update((int(np.argmax(values)), int(np.argmin(values))))
        with mp.workdps(80):
            for i in sorted(indices):
                exact_target = high_precision_value(blocks, coefficients, points[i])
                for name, run in runs.items():
                    exact = high_precision_value(run['blocks'], run['coefficients'], points[i])
                    worst = max(worst,
                                float(abs(mp.mpf(float(fields[name][i])) - exact)),
                                float(abs(mp.mpf(float(errors[name][i])) - abs(exact-exact_target))))
        planes.append(dict(second_coordinate=second+1, fields=fields, errors=errors,
                           validation_indices=sorted(int(i) for i in indices)))
    assert worst < 3e-13, worst

    # All panels share the same spatial limits; both planes share each color norm.
    solution_limit = 2.5
    error_floor, error_ceiling = 1e-9, 1e-6
    assert max(float(np.max(abs(p['fields']['target']))) for p in planes) < solution_limit
    assert max(float(np.max(e)) for p in planes for e in p['errors'].values()) < error_ceiling
    solution_cmap = plt.get_cmap('plasma')
    error_cmap = LinearSegmentedColormap.from_list(
        'fixed_blue_white_red', [style.BLUE, 'white', style.RED])
    field_norm = Normalize(-solution_limit, solution_limit)
    error_norm = LogNorm(error_floor, error_ceiling, clip=True)
    half_cell = .5 * (grid[1] - grid[0])
    extent = (grid[0]-half_cell, grid[-1]+half_cell,
              grid[0]-half_cell, grid[-1]+half_cell)
    fig = plt.figure(figsize=(5.35, 4.50))
    panel_width = .245
    panel_height = panel_width * 5.35 / 4.50
    axes = []
    images = []
    for row, plane in enumerate(planes):
        for column, name in enumerate(('target', 'NC', 'C')):
            ax = fig.add_axes((.087 + column*.318, .650 if row == 0 else .230,
                               panel_width, panel_height))
            axes.append(ax)
            values = (plane['fields']['target'] if column == 0 else plane['errors'][name])
            image = ax.imshow(np.asarray(values, float).reshape(s1.shape),
                              origin='lower', extent=extent,
                              norm=field_norm if column == 0 else error_norm,
                              cmap=solution_cmap if column == 0 else error_cmap,
                              interpolation='nearest', rasterized=True)
            images.append(image)
            if column == 0:
                ax.contour(s1, s2, np.asarray(values, float).reshape(s1.shape),
                           levels=[-2, -1, 0, 1, 2], colors=style.BLACK,
                           linewidths=.35, alpha=.25)
            ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
            ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
            ax.set_xlabel(r'$s_1$', labelpad=1)
            ax.set_ylabel(rf'$s_{plane["second_coordinate"]}$', labelpad=1)
            ax.tick_params(labelsize=7.5, pad=2, length=2.5)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(style.BLACK)
                spine.set_linewidth(.6)
            ax.set_title(f'({"abcdef"[3*row+column]}) ' +
                         ('Solution $u^*$' if column == 0 else f'Error: {name}'),
                         fontsize=8.4, pad=8)
            if column:
                ax.yaxis.label.set_visible(False)
                ax.tick_params(labelleft=False)
    style.square_panels(axes)
    field_bar = fig.colorbar(images[0], cax=fig.add_axes((.087, .097, .245, .018)),
                            orientation='horizontal', ticks=[-2, 0, 2])
    field_bar.set_label('Solution value', fontsize=7.5, labelpad=3)
    error_bar = fig.colorbar(images[1], cax=fig.add_axes((.405, .097, .563, .018)),
                            orientation='horizontal', ticks=[1e-9, 1e-8, 1e-7, 1e-6])
    error_bar.set_label(r'Pointwise error $|v-u^*|$ (shared log scale)', fontsize=7.5, labelpad=3)
    error_bar.ax.set_xticklabels([r'$\leq 10^{-9}$', r'$10^{-8}$', r'$10^{-7}$', r'$10^{-6}$'])
    for bar in (field_bar, error_bar):
        bar.ax.tick_params(labelsize=7.2, pad=2, length=2)
        bar.ax.xaxis.set_minor_locator(NullLocator())
        bar.outline.set_linewidth(.45)
    layout = layout_check(fig, axes)
    style.save(fig, HERE/STEM, dpi=600)
    shutil.copy2(HERE/f'{STEM}.pdf', ROOT/'paper_1111'/f'{STEM}.pdf')
    plt.close(fig)
    arrays = dict(grid=grid, origin=origin, principal_directions=basis[:3])
    for i, plane in enumerate(planes):
        for name, values in plane['fields'].items():
            arrays[f'plane_{i+1}_solution_{name}'] = np.asarray(values, float).reshape(s1.shape)
        for name, values in plane['errors'].items():
            arrays[f'plane_{i+1}_error_{name}'] = np.asarray(values, float).reshape(s1.shape)
    np.savez_compressed(HERE/f'{STEM}_fields.npz', **arrays)
    metadata = dict(
        description='Two principal solution slices and matched NC/C errors from saved first-order PB recovery states',
        example=dict(dimension=10, geometry=0, seed=10, sigma=SIGMA, reaction_strength=1,
                     selection='First seed of geometry 0 in dimension 10; no selection by final error',
                     method_mapping={'NC': 'NCx exact pair evaluator', 'C': 'T squared-gap cohort record'}),
        inputs={str(p.relative_to(ROOT)): digest(p) for p in PATHS.values()},
        scripts={str(p.relative_to(ROOT)): digest(p) for p in
                 (Path(__file__), HERE/'figure_style.py', HERE/'build_algorithm.py',
                  ROOT/'adaptive-desingularization/experiments/plot_style.py')},
        palette=list(style.PALETTE),
        plane_origin=origin.tolist(), principal_directions=basis[:3].tolist(),
        center_singular_values=singular_values.tolist(),
        plane_definition='x=origin+s1*q1+sj*qj; top j=2, bottom j=3; SVD directions of centered target centers, signed by largest coordinate',
        spatial_limits=[-1.3, 1.3], grid_size=len(grid),
        solution_limits=[-solution_limit, solution_limit], error_limits=[error_floor, error_ceiling],
        solution_colormap=dict(name='plasma'),
        error_scale='logarithmic, shared across all four error panels; values below 1e-9 mapped to blue',
        error_colormap=dict(name='fixed_blue_white_red', colors=[style.BLUE, '#FFFFFF', style.RED]),
        export_dpi=600,
        data_panel_borders=['left', 'right', 'bottom', 'top'],
        whole_space_relative_H1_error={name: run['final_relative_error'] for name, run in runs.items()},
        fits={name: run['fits'] for name, run in runs.items()},
        high_precision_check=dict(decimal_digits=80, maximum_absolute_discrepancy=worst,
                                  point_indices_by_plane=[p['validation_indices'] for p in planes]),
        maximum_slice_error=[{name:float(values.max()) for name, values in p['errors'].items()}
                             for p in planes],
        layout_check=layout, panel_geometry=style.panel_geometry(axes),
        outputs={ext: digest(HERE/f'{STEM}.{ext}') for ext in ('pdf', 'png', 'svg')},
        field_array_sha256=digest(HERE/f'{STEM}_fields.npz'))
    (HERE/f'{STEM}_provenance.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({key: metadata[key] for key in
                      ('whole_space_relative_H1_error', 'maximum_slice_error',
                       'high_precision_check', 'layout_check')}, indent=2))


if __name__ == '__main__':
    main()
