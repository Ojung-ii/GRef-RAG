"""Draw the manuscript's common-prompt context–F1 figure."""

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

DATASETS = (
    ("hotpotqa", "HotpotQA", (23, 42)),
    ("2wiki", "2Wiki", (5, 19)),
    ("musique", "MuSiQue", (1, 9.5)),
    ("popqa", "PopQA", (34, 53)),
)
METHODS = (
    ("bm25", "BM25", "#646971", "o"),
    ("dense-rag", "Dense RAG", "#955F51", "P"),
    ("raptor", "RAPTOR", "#8254A6", "v"),
    ("hipporag2", "HippoRAG2", "#0072B2", "^"),
    ("lightrag", "LightRAG", "#E69F00", "s"),
    ("kg2rag", "KG²RAG", "#D55E00", "D"),
    ("pathrag", "PathRAG", "#319E9C", "X"),
    ("gref-rag", "GRef-RAG", "#207543", "*"),
)


def plot_efficiency(source: Path, output: Path):
    with source.open(encoding="utf-8", newline="") as stream:
        rows = {
            (row["dataset"], row["method"]): (float(row["context_tokens"]), 100 * float(row["f1"]))
            for row in csv.DictReader(stream)
        }
    mpl.rcParams.update(
        {"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, panels = plt.subplots(2, 2, figsize=(7.0, 4.1), sharex=True)
    for ax, (dataset, label, ylim) in zip(panels.flat, DATASETS, strict=True):
        ax.set_title(label, loc="left", fontweight="bold", fontsize=9.5)
        ax.set_xscale("log")
        ax.set_xlim(410, 5200)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(FixedLocator([500, 1000, 2000, 4000]))
        ax.xaxis.set_major_formatter(FixedFormatter(["500", "1,000", "2,000", "4,000"]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(alpha=0.22)
        for key, name, color, marker in METHODS:
            x, y = rows[dataset, key]
            outlined = key == "dense-rag"
            ax.scatter(
                x,
                y,
                label=name,
                facecolor="none" if outlined else color,
                marker=marker,
                edgecolor=color if outlined else "white",
                linewidth=0.8 if outlined else 0.5,
                s=85 if outlined or key == "gref-rag" else 44,
                zorder=4 if outlined else 3,
            )
    handles, labels = panels[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.99),
        ncol=4,
        frameon=False,
        columnspacing=1.3,
    )
    fig.supxlabel("Mean retrieved-context tokens (log scale)", y=0.035)
    fig.supylabel("Answer F1 (%)", x=0.028)
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.13, top=0.79, wspace=0.17, hspace=0.42)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("res/common_prompt.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_efficiency(args.input, args.output)
