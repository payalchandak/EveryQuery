"""EveryQuery's AUROC gain over the autoregressive baseline, by task group and outcome prevalence.

Draws a two-panel dot plot, (a) ICU stay and (b) longitudinal care.  Each row is a task group and each
dot is the mean delta AUROC (EveryQuery minus autoregressive) over the group's tasks in one prevalence
bin (<1%, 1-5%, >=5%), with darker dots for rarer outcomes.  A gradient line joins a group's bins.  Rows
are ordered by the group's mean delta over all of its tasks, largest at the top.

Inputs are the per-query results CSVs (columns ``Query Specification``, ``Prevalence``,
``EveryQuery AUROC``, ``Autoregressive AUROC``), one per evaluation source, e.g.
``mimic_icu_24h_minpos40.csv``.  The source of a file is its stem with any ``_minposN`` suffix removed;
each (source, query) pair is looked up in ``task_groups.csv`` next to this script, which assigns its
setting and task group.  Every results row must be mapped, or the script stops and lists the missing ones.

Usage:
    python analysis/task_group_prevalence/plot_gain_by_prevalence.py \\
        mimic_icu_24h_minpos40.csv nwicu_icu_24h_minpos40.csv amc_outpatient_visit_general_minpos40.csv \\
        amc_outpatient_visit_hf_codes_minpos40.csv amc_outpatient_visit_t2d_codes_minpos40.csv

Writes ``eq_gain_by_group_prevalence.{pdf,png}`` to ``analysis/figures/`` unless ``--out-dir`` is given.
"""

import argparse
import re
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
MAPPING = HERE / "task_groups.csv"
DEFAULT_OUT_DIR = HERE.parent / "figures"

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BINS = ["<1%", "1\u20135%", "\u22655%"]  # en dash and greater-or-equal, for the legend
BIN_EDGES = [0.01, 0.05]
COLORS = dict(zip(BINS, ["#0d3a73", "#2a78d6", "#9cc5f0"], strict=True))  # one blue hue, dark = rare
PANELS = [("(a) ICU stay", "ICU"), ("(b) Longitudinal care", "longitudinal")]
X_LIM = (-0.2, 0.36)


def source_of(path: Path) -> str:
    """Return the evaluation source a results file belongs to.

    >>> source_of(Path("data/mimic_icu_24h_minpos40.csv"))
    'mimic_icu_24h'
    >>> source_of(Path("amc_outpatient_visit_general.csv"))
    'amc_outpatient_visit_general'
    """
    return re.sub(r"_minpos\d+$", "", path.stem)


def prevalence_bin(prevalence: pl.Expr) -> pl.Expr:
    """Label each prevalence with its bin: [0, 1%), [1%, 5%), [5%, 100%]."""
    return (
        pl.when(prevalence < BIN_EDGES[0])
        .then(pl.lit(BINS[0]))
        .when(prevalence < BIN_EDGES[1])
        .then(pl.lit(BINS[1]))
        .otherwise(pl.lit(BINS[2]))
    )


def load_results(paths: list[Path]) -> pl.DataFrame:
    """Read the results CSVs, attach setting and task group, and compute delta AUROC per task."""
    results = pl.concat(
        pl.read_csv(p).select(
            pl.lit(source_of(p)).alias("source"),
            pl.col("Query Specification").alias("query"),
            pl.col("Prevalence").alias("prevalence"),
            (pl.col("EveryQuery AUROC") - pl.col("Autoregressive AUROC")).alias("delta"),
        )
        for p in paths
    )
    mapped = results.join(pl.read_csv(MAPPING), on=["source", "query"], how="left")
    missing = mapped.filter(pl.col("task_group").is_null())
    if missing.height:
        rows = "\n".join(f"  {s}: {q}" for s, q in missing.select("source", "query").iter_rows())
        raise ValueError(f"{missing.height} results rows have no entry in {MAPPING.name}:\n{rows}")
    return mapped.with_columns(prevalence_bin(pl.col("prevalence")).alias("bin"))


def gradient_line(ax: plt.Axes, x0: float, x1: float, y: float, c0: str, c1: str, n: int = 40) -> None:
    """Draw a horizontal line from x0 to x1 whose color blends from c0 to c1."""
    xs = np.linspace(x0, x1, n + 1)
    segments = np.stack([np.c_[xs[:-1], np.full(n, y)], np.c_[xs[1:], np.full(n, y)]], axis=1)
    colors = [np.array(to_rgb(c0)) * (1 - t) + np.array(to_rgb(c1)) * t for t in np.linspace(0, 1, n)]
    ax.add_collection(
        LineCollection(segments, colors=colors, lw=3, capstyle="projecting", antialiased=False, zorder=2)
    )


def draw_panel(ax: plt.Axes, tasks: pl.DataFrame) -> None:
    """Plot one setting: a row per task group, a dot per prevalence bin."""
    order = (
        tasks.group_by("task_group")
        .agg(pl.col("delta").mean())
        .sort(["delta", "task_group"], descending=[True, False])
        .get_column("task_group")
        .to_list()
    )
    by_bin = tasks.group_by("task_group", "bin").agg(pl.col("delta").mean())
    for y, group in enumerate(order):
        ax.axhline(y, color=GRID, lw=0.8, zorder=0)
        means = dict(by_bin.filter(pl.col("task_group") == group).select("bin", "delta").iter_rows())
        points = [(b, means[b]) for b in BINS if b in means]
        for (b0, x0), (b1, x1) in pairwise(points):
            gradient_line(ax, x0, x1, y, COLORS[b0], COLORS[b1])
        for b, x in points:
            z = 3 + len(BINS) - BINS.index(b)  # rarer bins are drawn on top where dots overlap
            ax.scatter(x, y, s=60, color=COLORS[b], edgecolor="white", lw=1.0, zorder=z)
    ax.axvline(0, color=INK, lw=1, zorder=1)
    ax.set_yticks(range(len(order)), order)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlim(*X_LIM)
    ax.set_xlabel("\u0394 AUROC  (EveryQuery \u2212 Autoregressive)")
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    handles = [Line2D([], [], marker="o", ls="", color=COLORS[b], ms=7, mec="white", label=b) for b in BINS]
    legend = ax.legend(
        handles=handles,
        title="Prevalence",
        title_fontsize=8.5,
        fontsize=8.5,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1,
        borderaxespad=0.3,
        handletextpad=0.3,
        labelspacing=0.35,
        alignment="left",
    )
    legend.set_zorder(20)


def plot(tasks: pl.DataFrame) -> plt.Figure:
    """Build the two-panel figure."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), sharex=True, gridspec_kw={"wspace": 0.62})
    for ax, (_, setting) in zip(axes, PANELS, strict=True):
        draw_panel(ax, tasks.filter(pl.col("setting") == setting))

    # Left-align each panel title with the left edge of its task labels, not the plot area.
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    for ax, (title, _) in zip(axes, PANELS, strict=True):
        left = to_fig.transform_bbox(ax.get_tightbbox(fig.canvas.get_renderer())).x0
        top = to_fig.transform(ax.transAxes.transform((0, 1)))[1]
        fig.text(left, top + 0.015, title, ha="left", va="bottom", fontweight="bold", fontsize=10)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", nargs="+", type=Path, help="per-query results CSVs, one per source")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="where to write the figure")
    args = parser.parse_args()

    fig = plot(load_results(args.results))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = args.out_dir / f"eq_gain_by_group_prevalence.{ext}"
        fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
