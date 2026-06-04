"""Phase 4 matplotlib figures — six PNG plots driven by Phase 3/4 dataframes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = {
    "navy": "#1A2438",
    "amber": "#D97706",
    "teal": "#0F766E",
    "slate": "#64748B",
    "red": "#B91C1C",
}

FAMILY_COLOR = {
    "classical_ml": PALETTE["slate"],
    "roberta": PALETTE["navy"],
    "llama": PALETTE["teal"],
    "qwen": PALETTE["amber"],
    "gpt_oss": PALETTE["red"],
}

DATASET_COLOR = {
    "fakenewsnet": PALETTE["teal"],
    "employee_reviews": PALETTE["amber"],
}

PROMPT_COLOR = {
    "ZS": PALETTE["slate"],
    "ZS_CoT": PALETTE["teal"],
    "FS_CoT_RP_NA": PALETTE["amber"],
}

DATASET_LABEL = {
    "fakenewsnet": "FakeNewsNet (binary)",
    "employee_reviews": "Employee Reviews (3-class)",
}

FIG_DPI = 150
FIG_SIZE = (10, 6)
FIG_SIZE_TWOPANEL = (12, 5)


def _apply_house_style(ax) -> None:
    """Apply consistent grid/spines styling to an Axes.

    Args:
        ax: Matplotlib Axes to style.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> None:
    """Save figure to PNG at FIG_DPI then close it.

    Args:
        fig: Matplotlib Figure.
        path: Destination PNG path (parent dirs created if needed).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def fig1_f1_master(by_model: pd.DataFrame, out: Path) -> None:
    """Two-panel horizontal bar chart: best F1 per model on each dataset.

    Args:
        by_model: Output of ``aggregation.best_per_model``.
        out: Destination PNG path.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, sharex=True)
    for ax, dataset in zip(axes, ["fakenewsnet", "employee_reviews"], strict=False):
        sub = (
            by_model[by_model["dataset"] == dataset]
            .sort_values("f1_weighted", ascending=True)
            .reset_index(drop=True)
        )
        colors = [FAMILY_COLOR.get(f, PALETTE["slate"]) for f in sub["model_family"]]
        bars = ax.barh(sub["model"], sub["f1_weighted"], color=colors, alpha=0.9)
        for bar, val in zip(bars, sub["f1_weighted"], strict=False):
            ax.text(
                val + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                ha="left",
                fontsize=8,
            )
        baselines_sub = sub[sub["source"] == "baseline"]
        if not baselines_sub.empty:
            best_baseline_row = baselines_sub.loc[baselines_sub["f1_weighted"].idxmax()]
            best_baseline_f1 = float(best_baseline_row["f1_weighted"])
            ax.axvline(
                best_baseline_f1,
                color=PALETTE["navy"],
                linestyle=":",
                linewidth=1.2,
                alpha=0.7,
                label=f"best baseline = {best_baseline_f1:.3f}",
            )
            ax.legend(loc="lower right", fontsize=8, frameon=False)
        ax.set_title(DATASET_LABEL[dataset], fontsize=11)
        ax.set_xlabel("Weighted F1")
        ax.set_xlim(0.0, 1.0)
        _apply_house_style(ax)
    fig.suptitle("Weighted F1 — 210 FNN + 204 ER test records", fontsize=13, y=1.02)
    _save(fig, out)


def _pareto_frontier_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return Pareto frontier (low x, high y) of the given point list.

    Args:
        points: (x, y) pairs where minimising x and maximising y are both better.

    Returns:
        Sorted-by-x frontier points (left-to-right, monotonically decreasing y).
    """
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    frontier: list[tuple[float, float]] = []
    best_y = -float("inf")
    for x, y in reversed(pts):
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    return sorted(frontier, key=lambda p: p[0])


def fig2_pareto(combined: pd.DataFrame, out: Path) -> None:
    """Pareto plot of all 32 combined rows: F1 vs median latency (log-x).

    Args:
        combined: Output of ``aggregation.combine`` — 32 rows.
        out: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for source, marker in [("baseline", "s"), ("llm", "o")]:
        sub = combined[combined["source"] == source]
        colors = [FAMILY_COLOR.get(f, PALETTE["slate"]) for f in sub["model_family"]]
        ax.scatter(
            sub["median_latency_ms"],
            sub["f1_weighted"],
            c=colors,
            marker=marker,
            s=90,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8,
        )
    # Label only a curated subset to avoid label collisions in the upper-right
    # LLM cluster. A point gets a label iff any of:
    #   * f1_weighted < 0.78 (low-F1 outliers)
    #   * median_latency_ms < 1000 (fast baselines)
    #   * model_family == "classical_ml" (cheap classical-ML row)
    #   * it is the top-1 F1 within its (family, dataset) group
    top1_idx = set(
        combined.groupby(["model_family", "dataset"])["f1_weighted"].idxmax().tolist()
    )
    for i, row in combined.iterrows():
        should_label = (
            row["f1_weighted"] < 0.78
            or row["median_latency_ms"] < 1000
            or row["model_family"] == "classical_ml"
            or i in top1_idx
        )
        if not should_label:
            continue
        ax.annotate(
            f"{row['model']}\n({row['dataset'][:3]})",
            xy=(row["median_latency_ms"], row["f1_weighted"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=7,
            color="#222",
        )
    # Pareto frontier over ALL 32 points.
    pts = list(zip(combined["median_latency_ms"], combined["f1_weighted"], strict=False))
    frontier = _pareto_frontier_points(pts)
    if frontier:
        fx = [p[0] for p in frontier]
        fy = [p[1] for p in frontier]
        ax.plot(fx, fy, color="#888", linestyle="--", linewidth=1.0, alpha=0.8, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Median latency per record (ms, log scale)")
    ax.set_ylabel("Weighted F1")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Pareto — F1 vs latency (all 32 combos)")

    # Custom legend: family colors + shape semantics.
    handles = []
    for family, color in FAMILY_COLOR.items():
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=8,
                label=family,
            )
        )
    handles.append(
        plt.Line2D(
            [0], [0], marker="s", color="w", markerfacecolor="#444", markersize=8, label="baseline"
        )
    )
    handles.append(
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#444", markersize=8, label="LLM"
        )
    )
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    _apply_house_style(ax)
    fig.text(
        0.02,
        -0.02,
        "32 points = 10 baselines + 22 LLM combos. Labels mark best prompt per "
        "(model, dataset); other prompts visible without labels. Baseline latency = "
        "mean-per-record (CV); LLM = observed median per call (Groq).",
        fontsize=7,
        color="#555",
        wrap=True,
    )
    _save(fig, out)


def fig3_axis_a_scaling(llm_summary_df: pd.DataFrame, out: Path) -> None:
    """Axis A — size scaling (Llama 8B vs 70B) + Qwen3 reasoning toggle.

    Args:
        llm_summary_df: Output of ``aggregation.llm_summary``.
        out: Destination PNG path.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_TWOPANEL)

    # Left: size scaling — best F1 across prompts per (model, dataset).
    ax_l = axes[0]
    sizes = {"llama-3.1-8b-instant": 8, "llama-3.3-70b-versatile": 70}
    for dataset, color in DATASET_COLOR.items():
        xs: list[int] = []
        ys: list[float] = []
        prompts: list[str] = []
        for slug, size in sizes.items():
            sub = llm_summary_df[
                (llm_summary_df["logical_name"] == slug) & (llm_summary_df["dataset"] == dataset)
            ]
            if sub.empty:
                continue
            row = sub.loc[sub["f1_weighted"].idxmax()]
            xs.append(size)
            ys.append(float(row["f1_weighted"]))
            prompts.append(str(row["prompt_code"]))
        ax_l.plot(xs, ys, marker="o", color=color, linewidth=2, label=DATASET_LABEL[dataset])
        for x, y, p in zip(xs, ys, prompts, strict=False):
            ax_l.annotate(
                f"{y:.3f}\n{p}",
                xy=(x, y),
                xytext=(6, -4),
                textcoords="offset points",
                fontsize=7,
            )
    ax_l.set_xscale("log")
    ax_l.set_xticks([8, 70])
    ax_l.set_xticklabels(["8B", "70B"])
    ax_l.set_xlabel("Llama model size")
    ax_l.set_ylabel("Weighted F1 (best prompt)")
    ax_l.set_ylim(0.6, 0.95)
    ax_l.set_title("Llama 3.1 8B vs 3.3 70B (best prompt)")
    ax_l.legend(loc="lower right", fontsize=8, frameon=False)
    _apply_house_style(ax_l)

    # Right: Qwen3 reasoning toggle (ZS only).
    ax_r = axes[1]
    efforts = ["none", "default"]
    width = 0.35
    x = np.arange(len(efforts))
    for i, (dataset, color) in enumerate(DATASET_COLOR.items()):
        f1s: list[float] = []
        for effort in efforts:
            slug = f"qwen3-32b-reasoning-{effort}"
            sub = llm_summary_df[
                (llm_summary_df["logical_name"] == slug)
                & (llm_summary_df["dataset"] == dataset)
                & (llm_summary_df["prompt_code"] == "ZS")
            ]
            f1s.append(float(sub["f1_weighted"].iloc[0]) if not sub.empty else float("nan"))
        offset = (i - 0.5) * width
        bars = ax_r.bar(
            x + offset, f1s, width=width, color=color, alpha=0.9, label=DATASET_LABEL[dataset]
        )
        for bar, val in zip(bars, f1s, strict=False):
            if not np.isnan(val):
                ax_r.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.005,
                    f"{val:.3f}",
                    ha="center",
                    fontsize=7,
                )
        if all(not np.isnan(v) for v in f1s):
            delta = f1s[1] - f1s[0]
            ax_r.annotate(
                f"Δ={delta:+.3f}",
                xy=(x[1] + offset, f1s[1]),
                xytext=(0, 18),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="#444",
                arrowprops={"arrowstyle": "->", "color": "#999", "lw": 0.6},
            )
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(["reasoning=none", "reasoning=default"])
    ax_r.set_ylabel("Weighted F1 (ZS)")
    ax_r.set_ylim(0.6, 0.95)
    ax_r.set_title("Qwen3 32B reasoning toggle (ZS)")
    ax_r.legend(loc="lower right", fontsize=8, frameon=False)
    _apply_house_style(ax_r)

    fig.suptitle("Axis A — Scaling and reasoning toggle (Qwen3 32B, ZS only)", fontsize=12)
    _save(fig, out)


def fig4_axis_b_reasoning(llm_summary_df: pd.DataFrame, out: Path) -> None:
    """Axis B — GPT-OSS vs Llama 3.3 70B head-to-head + reasoning-token scatter.

    Args:
        llm_summary_df: Output of ``aggregation.llm_summary``.
        out: Destination PNG path.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE_TWOPANEL)

    # Left: grouped bars per dataset x prompt (GPT-OSS vs Llama 3.3 70B).
    ax_l = axes[0]
    prompts = ["ZS", "ZS_CoT", "FS_CoT_RP_NA"]
    datasets = ["fakenewsnet", "employee_reviews"]
    width = 0.18
    x_positions = np.arange(len(prompts))
    # 2 datasets x 2 models = 4 bars per prompt.
    for d_i, dataset in enumerate(datasets):
        for m_i, (slug, hatch) in enumerate(
            [("openai-gpt-oss-120b", ""), ("llama-3.3-70b-versatile", "//")]
        ):
            f1s: list[float] = []
            for p in prompts:
                sub = llm_summary_df[
                    (llm_summary_df["logical_name"] == slug)
                    & (llm_summary_df["dataset"] == dataset)
                    & (llm_summary_df["prompt_code"] == p)
                ]
                f1s.append(float(sub["f1_weighted"].iloc[0]) if not sub.empty else float("nan"))
            offset = (d_i * 2 + m_i - 1.5) * width
            color = DATASET_COLOR[dataset]
            label_short = "GPT-OSS" if slug == "openai-gpt-oss-120b" else "Llama 70B"
            label = f"{label_short} / {dataset[:3]}"
            bars = ax_l.bar(
                x_positions + offset,
                f1s,
                width=width,
                color=color,
                alpha=0.9,
                hatch=hatch,
                edgecolor="white",
                label=label,
            )
            for bar, val in zip(bars, f1s, strict=False):
                if not np.isnan(val):
                    ax_l.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + 0.005,
                        f"{val:.2f}",
                        ha="center",
                        fontsize=6,
                    )
    ax_l.set_xticks(x_positions)
    ax_l.set_xticklabels(prompts, fontsize=8)
    ax_l.set_ylabel("Weighted F1")
    ax_l.set_ylim(0.5, 0.95)
    ax_l.set_title("GPT-OSS 120B vs Llama 3.3 70B per prompt")
    ax_l.legend(loc="lower right", fontsize=6, frameon=False, ncol=2)
    _apply_house_style(ax_l)

    # Right: scatter of total reasoning_tokens vs F1.
    ax_r = axes[1]
    reasoning_slugs = ["openai-gpt-oss-120b", "qwen3-32b-reasoning-default"]
    sub = llm_summary_df[llm_summary_df["logical_name"].isin(reasoning_slugs)]
    for dataset, color in DATASET_COLOR.items():
        sub_d = sub[sub["dataset"] == dataset]
        ax_r.scatter(
            sub_d["total_reasoning_tokens"],
            sub_d["f1_weighted"],
            c=color,
            marker="o",
            s=90,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.8,
            label=DATASET_LABEL[dataset],
        )
        for r in sub_d.itertuples():
            short = "GPT-OSS" if r.logical_name == "openai-gpt-oss-120b" else "Qwen3-d"
            ax_r.annotate(
                f"{short} {r.prompt_code}",
                xy=(r.total_reasoning_tokens, r.f1_weighted),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=6,
                color="#333",
            )
    ax_r.set_xscale("log")
    ax_r.set_xlabel("Total reasoning tokens per combo (log)")
    ax_r.set_ylabel("Weighted F1")
    ax_r.set_ylim(0.5, 0.95)
    ax_r.set_title("Reasoning tokens vs F1")
    ax_r.legend(loc="lower right", fontsize=7, frameon=False)
    _apply_house_style(ax_r)

    fig.suptitle("Axis B — Reasoning model behaviour (GPT-OSS, Qwen3-default)", fontsize=12)
    _save(fig, out)


def fig5_prompt_effect(llm_summary_df: pd.DataFrame, out: Path) -> None:
    """Two-panel grouped bars: per-LLM F1 colored by prompt code.

    Args:
        llm_summary_df: Output of ``aggregation.llm_summary``.
        out: Destination PNG path.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "qwen3-32b-reasoning-none",
        "qwen3-32b-reasoning-default",
        "openai-gpt-oss-120b",
    ]
    short_names = {
        "llama-3.1-8b-instant": "Llama 3.1 8B",
        "llama-3.3-70b-versatile": "Llama 3.3 70B",
        "qwen3-32b-reasoning-none": "Qwen3 32B (none)",
        "qwen3-32b-reasoning-default": "Qwen3 32B (default)",
        "openai-gpt-oss-120b": "GPT-OSS 120B",
    }
    prompts = ["ZS", "ZS_CoT", "FS_CoT_RP_NA"]
    width = 0.25
    x = np.arange(len(models))
    for ax, dataset in zip(axes, ["fakenewsnet", "employee_reviews"], strict=False):
        for p_i, prompt in enumerate(prompts):
            f1s: list[float] = []
            for slug in models:
                sub = llm_summary_df[
                    (llm_summary_df["logical_name"] == slug)
                    & (llm_summary_df["dataset"] == dataset)
                    & (llm_summary_df["prompt_code"] == prompt)
                ]
                f1s.append(float(sub["f1_weighted"].iloc[0]) if not sub.empty else float("nan"))
            offset = (p_i - 1) * width
            colors = ["#cccccc" if np.isnan(v) else PROMPT_COLOR[prompt] for v in f1s]
            heights = [0.02 if np.isnan(v) else v for v in f1s]
            bars = ax.bar(
                x + offset,
                heights,
                width=width,
                color=colors,
                alpha=0.9,
                edgecolor="white",
                label=prompt,
            )
            for bar, val in zip(bars, f1s, strict=False):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + 0.005,
                        f"{val:.2f}",
                        ha="center",
                        fontsize=6,
                    )
                else:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        0.03,
                        "n/a",
                        ha="center",
                        fontsize=6,
                        color="#666",
                    )
        ax.set_xticks(x)
        ax.set_xticklabels([short_names[m] for m in models], fontsize=8, rotation=15)
        ax.set_ylabel("Weighted F1")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(DATASET_LABEL[dataset], fontsize=11)
        ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=3)
        _apply_house_style(ax)
    fig.suptitle("Prompt effect per model — FS_CoT_RP_NA size dependency", fontsize=12)
    _save(fig, out)


def fig6_reasoning_cost(llm_summary_df: pd.DataFrame, out: Path) -> None:
    """Scatter: median reasoning tokens per call vs F1, with zero-reasoning peers.

    Args:
        llm_summary_df: Output of ``aggregation.llm_summary``.
        out: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    reasoning_slugs = {"openai-gpt-oss-120b", "qwen3-32b-reasoning-default"}
    peer_slugs = {"llama-3.3-70b-versatile", "qwen3-32b-reasoning-none"}
    short_names = {
        "openai-gpt-oss-120b": "GPT-OSS",
        "qwen3-32b-reasoning-default": "Qwen3-d",
        "llama-3.3-70b-versatile": "Llama70",
        "qwen3-32b-reasoning-none": "Qwen3-n",
    }

    for dataset, color in DATASET_COLOR.items():
        sub_d = llm_summary_df[
            (llm_summary_df["logical_name"].isin(reasoning_slugs))
            & (llm_summary_df["dataset"] == dataset)
        ]
        ax.scatter(
            sub_d["median_reasoning_tokens"].clip(lower=1),
            sub_d["f1_weighted"],
            c=color,
            marker="o",
            s=110,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.8,
            label=f"{DATASET_LABEL[dataset]} (reasoning)",
        )
        for r in sub_d.itertuples():
            label = f"{short_names[r.logical_name]} {r.prompt_code}"
            x_pos = max(int(r.median_reasoning_tokens), 1)
            ax.annotate(
                label,
                xy=(x_pos, r.f1_weighted),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=7,
                color="#222",
            )

        # Zero-reasoning peers as crosses at left edge.
        sub_p = llm_summary_df[
            (llm_summary_df["logical_name"].isin(peer_slugs))
            & (llm_summary_df["dataset"] == dataset)
        ]
        ax.scatter(
            [0.5] * len(sub_p),
            sub_p["f1_weighted"],
            c=color,
            marker="x",
            s=70,
            alpha=0.55,
            linewidths=1.5,
            label=f"{DATASET_LABEL[dataset]} (peer, 0 tokens)",
        )
        for r in sub_p.itertuples():
            label = f"{short_names[r.logical_name]} {r.prompt_code}"
            ax.annotate(
                label,
                xy=(0.5, r.f1_weighted),
                xytext=(8, 2),
                textcoords="offset points",
                fontsize=6,
                color="#555",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Median reasoning tokens per call (log; peers at 0 plotted at 0.5)")
    ax.set_ylabel("Weighted F1")
    ax.set_ylim(0.5, 0.95)
    ax.set_title("Reasoning cost-effectiveness — F1 per reasoning token")
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    _apply_house_style(ax)
    _save(fig, out)
