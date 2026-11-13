"""Shared four-color style with square data panels and text outside data axes."""
import json
import math
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / 'adaptive-desingularization/experiments/plot_style.py').exists())
sys.path.insert(0, str(ROOT / 'adaptive-desingularization/experiments'))
import plot_style as _base

BLUE = '#0000FF'
RED = '#FF0000'
BLACK = '#111111'
GREEN = '#188038'
PALETTE = (BLUE, RED, BLACK, GREEN)


def apply_style(size=8.0):
    _base.apply_style(size)
    import matplotlib as mpl
    from cycler import cycler
    mpl.rcParams.update({
        'axes.prop_cycle': cycler(color=PALETTE),
        'axes.edgecolor': BLACK, 'axes.labelcolor': BLACK,
        'axes.titlecolor': BLACK, 'text.color': BLACK,
        'xtick.color': BLACK, 'ytick.color': BLACK,
        'axes.spines.top': False, 'axes.spines.right': False,
        'grid.color': BLACK, 'grid.alpha': 0.10, 'grid.linewidth': 0.45,
        'legend.frameon': False, 'axes.titlepad': 9,
    })


def square_panels(axes):
    """Set the physical height/width ratio without changing the data limits."""
    for ax in axes:
        ax.set_box_aspect(1)


def panel_geometry(axes):
    """Measure the rendered data rectangles, excluding labels and legends."""
    axes = list(axes)
    if not axes:
        return []
    fig = axes[0].figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return [dict(title=ax.get_title(),
                 width_inches=ax.get_window_extent(renderer).width / fig.dpi,
                 height_inches=ax.get_window_extent(renderer).height / fig.dpi,
                 height_over_width=ax.get_window_extent(renderer).height /
                 ax.get_window_extent(renderer).width)
            for ax in axes]


def save(fig, stem, *, dpi=None):
    """Export with an optional resolution for embedded raster fields and PNG."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    data_axes = [ax for ax in fig.axes if ax.get_label() != '<colorbar>'
                 and not hasattr(ax, '_colorbar')]
    geometry = panel_geometry(data_axes)
    assert all(math.isclose(panel['height_over_width'], 1., abs_tol=1e-8)
               for panel in geometry), f'Non-square data panel in {stem.name}: {geometry}'
    for ext in ('pdf', 'svg', 'png'):
        opts = dict(bbox_inches='tight', pad_inches=0.06)
        if dpi is not None:
            opts['dpi'] = dpi
        elif ext == 'png':
            opts['dpi'] = 220
        if ext == 'pdf': opts['metadata'] = {'Creator': 'Matplotlib; saved-data publication figures'}
        fig.savefig(stem.with_suffix('.' + ext), **opts)
    stem.with_name(stem.name + '_layout.json').write_text(
        json.dumps(dict(panel_shape='1:1', data_panels=geometry), indent=2) + '\n')

