"""Publication plotting defaults, preserving the palette of the companion paper's figures.

The colors and serif/Computer Modern typography follow the collision and
derivative-completion figures of the singular-representations repository.
Import this module before pyplot, call ``apply_style()``, and use
``save_figure(fig, stem)`` for vector PDF/SVG plus a 600-dpi PNG. All
quantities in these figures must come from experiment records.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

# Permit reproducible, headless builds in a restricted workspace.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "desingularization-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "desingularization-cache"))

import matplotlib as mpl
from matplotlib import font_manager

mpl.use("Agg")

BLUE = "#0000FF"
RED = "#FF0000"
GREEN = "#188038"
ORANGE = "#D55E00"
PURPLE = "#7B3294"
GOLD = "#E69F00"
BLACK = "#111111"
GRAY = "#666666"
LIGHT_GRAY = "#CFCFCF"
PALE_GRAY = "#F1F1F1"
ORDER_COLORS = {0: BLUE, 1: ORANGE, 2: PURPLE, 3: GREEN}
METHOD_COLORS = {"standard": BLUE, "ablation": GREEN, "adaptive": RED}
METHOD_STYLES = {"standard": "-", "ablation": "-.", "adaptive": "--"}
METHOD_LABELS = {"standard": "Standard optimization", "ablation": "Stable coordinates only", "adaptive": "Adaptive completion"}
FULL_WIDTH = 7.15
COMPACT_WIDTH = 4.685

# Latin Modern Roman as shipped with TeX Live, when matplotlib has not registered it itself.
_LM_DIRS = (
    "/usr/local/texlive/2025/texmf-dist/fonts/opentype/public/lm",
    "/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/lm",
    "/usr/share/texmf/fonts/opentype/public/lm",
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm",
)


def _register_latin_modern() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    if "Latin Modern Roman" in names:
        return
    for directory in _LM_DIRS:
        for stem in ("lmroman10-regular", "lmroman10-italic", "lmroman10-bold", "lmroman10-bolditalic"):
            path = Path(directory) / f"{stem}.otf"
            if path.exists():
                try:
                    font_manager.fontManager.addfont(str(path))
                except Exception:  # pragma: no cover - a missing or broken font file only changes typography
                    pass


def apply_style(font_size: float = 9.0) -> None:
    """Use the companion palette with text sized for a full-width paper figure."""
    _register_latin_modern()
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "Computer Modern Roman", "DejaVu Serif"],
        "font.size": font_size,
        "mathtext.fontset": "cm",
        "text.usetex": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.75,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 0.5,
        "axes.titleweight": "normal",
        "axes.titlecolor": "black",
        "axes.titlepad": 7.0,
        "axes.labelpad": 3.0,
        "axes.grid": False,
        "axes.axisbelow": True,
        "xtick.labelsize": font_size - 1.0,
        "ytick.labelsize": font_size - 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "lines.linewidth": 1.5,
        "lines.markersize": 3.0,
        "legend.fontsize": font_size - 1.0,
        "legend.frameon": False,
        "legend.handlelength": 2.4,
        "legend.borderaxespad": 0.4,
        "legend.labelspacing": 0.3,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 600,
        "savefig.pad_inches": 0.04,
    })


def panel_title(axis, letter: str, title: str) -> None:
    """Add the manuscript's '(a) description' panel title."""
    axis.set_title(f"({letter}) {title}")


def save_figure(figure, stem: str | Path, *, dpi: int = 600) -> dict[str, str]:
    """Export a tightly cropped vector master and high-resolution preview."""
    path = Path(stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for extension in ("pdf", "svg", "png"):
        destination = path.with_suffix(f".{extension}")
        kwargs = {"bbox_inches": "tight", "dpi": dpi}
        if extension == "pdf":
            kwargs["metadata"] = {"Creator": "Matplotlib; adaptive desingularization experiment scripts"}
        figure.savefig(destination, **kwargs)
        outputs[extension] = str(destination)
    return outputs
