from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image

from .io_utils import ROOT, write_csv


RUN_DIR = ROOT / "03_逐运行结果"
TABLE_DIR = ROOT / "04_汇总表格"
FIGURE_ROOT = ROOT / "05_论文图表"
AUDIT_DIR = ROOT / "06_审计与复现"

PALETTE = {
    "ink": "#24303A",
    "blue": "#4C78A8",
    "teal": "#3F8F83",
    "coral": "#D9785F",
    "gold": "#D8A83E",
    "gray": "#A9B0B6",
    "light": "#EEF1F3",
    "red": "#B84A4A",
}
MODEL_COLORS = {
    "count": PALETTE["gray"],
    "deepseek": PALETTE["coral"],
    "linear": PALETTE["blue"],
    "tree": PALETTE["teal"],
}
MODE_COLORS = {"hard": PALETTE["blue"], "soft": PALETTE["gold"], "none": PALETTE["coral"]}


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 8.0,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.3,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def _save_all(fig: plt.Figure, stem: str) -> None:
    destinations = {
        "PNG": (".png", 400),
        "PDF": (".pdf", None),
        "SVG": (".svg", None),
        "TIFF": (".tiff", 400),
    }
    for directory, (suffix, dpi) in destinations.items():
        output_dir = FIGURE_ROOT / directory
        output_dir.mkdir(parents=True, exist_ok=True)
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if dpi is not None:
            kwargs["dpi"] = dpi
        if directory == "TIFF":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(output_dir / f"{stem}{suffix}", **kwargs)
    plt.close(fig)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str) -> None:
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, linewidth=1.3, color=color)
    )


def figure_1() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.5), gridspec_kw={"width_ratios": [1.18, 1]})
    ax, eq = axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    _panel_label(ax, "a")
    ax.set_title("Wireless edge offloading system", loc="left", pad=8)

    edge = FancyBboxPatch((6.4, 3.2), 2.8, 3.6, boxstyle="round,pad=0.12", fc="#E8F0F7", ec=PALETTE["blue"], lw=1.2)
    ax.add_patch(edge)
    ax.text(7.8, 5.8, "Edge GPU pool", ha="center", va="center", fontweight="bold", color=PALETTE["blue"])
    ax.text(7.8, 5.05, "shared compute", ha="center", va="center")
    ax.text(7.8, 4.55, "+ memory", ha="center", va="center")
    ax.text(7.8, 3.85, "controller", ha="center", va="center", color=PALETTE["ink"])

    positions = [(1.2, 7.8), (1.1, 5.0), (1.5, 2.0), (3.5, 8.5), (3.8, 1.6)]
    labels = [r"$i=1$", r"$i=2$", r"$i=3$", r"$\cdots$", r"$i=N$"]
    for (x, y), label in zip(positions, labels):
        ax.add_patch(Circle((x, y), 0.55, fc="white", ec=PALETTE["ink"], lw=1.0))
        ax.text(x, y, label, ha="center", va="center")
        _arrow(ax, (x + 0.55, y), (6.25, 5.0), PALETTE["teal"])
    ax.text(3.7, 6.0, "uplink intent + payload", ha="center", color=PALETTE["teal"])
    ax.text(3.8, 4.2, "decision + downlink output", ha="center", color=PALETTE["blue"])
    _arrow(ax, (6.25, 4.7), (2.0, 4.1), PALETTE["blue"])
    ax.text(0.45, 0.45, r"Each node chooses $s_i\in\{0,1\}$", color=PALETTE["ink"])

    eq.axis("off")
    _panel_label(eq, "b")
    eq.set_title("Coupled state and hard constraints", loc="left", pad=8)
    equations = [
        (0.95, r"$K(\mathbf{s})=\sum_i s_i$", "offloaded task count", PALETTE["gray"]),
        (0.76, r"$W(\mathbf{s})=\sum_i s_i q_i$", "aggregate workload", PALETTE["teal"]),
        (0.57, r"$V(\mathbf{s})=\sum_i s_i v_i$", "aggregate memory", PALETTE["coral"]),
    ]
    for y, formula, note, color in equations:
        eq.add_patch(FancyBboxPatch((0.04, y - 0.115), 0.92, 0.14, boxstyle="round,pad=0.02", fc="white", ec=color, lw=1.0, transform=eq.transAxes))
        eq.text(0.09, y - 0.03, formula, transform=eq.transAxes, fontsize=10, color=PALETTE["ink"])
        eq.text(0.09, y - 0.085, note, transform=eq.transAxes, color=color)
    eq.text(0.08, 0.36, r"Feasible: $W\leq W_{\max}$ and $V\leq V_{\max}$", transform=eq.transAxes, fontsize=8, fontweight="bold")
    eq.text(0.08, 0.25, r"Congestion: $D_{\rm comp}=D_0+a\,\rho/(1-\rho)$, $\rho=W/C_W$", transform=eq.transAxes, fontsize=8)
    eq.text(0.08, 0.12, "WA-MCBR accepts only feasible single flips\nthat reduce the public objective J.", transform=eq.transAxes, linespacing=1.5)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.91, bottom=0.06, wspace=0.15)
    _save_all(fig, "Fig_1_system_model")


def figure_2() -> None:
    predictions = pd.read_csv(RUN_DIR / "profiler_predictions.csv")
    profiles = pd.read_csv(ROOT / "02_任务池与画像" / "task_profiles.csv")
    metrics = pd.read_csv(TABLE_DIR / "table_ii_profiler_metrics.csv").set_index("model")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    ax = axes[0]
    _panel_label(ax, "a")
    for model in ["count", "deepseek", "linear", "tree"]:
        part = predictions.loc[predictions.model.eq(model)]
        label = f"{model.capitalize()}  R²={metrics.loc[model, 'r2']:.2f}"
        ax.scatter(part.q_true_data, part.q_predicted, s=9, alpha=0.34, color=MODEL_COLORS[model], label=label, edgecolors="none")
    limit = float(np.nanpercentile(np.r_[predictions.q_true_data, predictions.q_predicted], 99.5))
    ax.plot([0, limit], [0, limit], color=PALETTE["ink"], lw=1.0, ls="--")
    ax.set(xlabel="Observed normalized workload", ylabel="Predicted normalized workload", xlim=(0, limit), ylim=(0, limit), title="Held-out workload prediction")
    ax.legend(loc="upper left", ncol=1)

    ax = axes[1]
    _panel_label(ax, "b")
    for column, label, color in [
        ("q_count", "Count", MODEL_COLORS["count"]),
        ("q_llm", "DeepSeek", MODEL_COLORS["deepseek"]),
        ("q_data", "Observed data", PALETTE["ink"]),
    ]:
        values = np.sort(profiles[column].to_numpy(float))
        cdf = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, cdf, label=label, color=color)
    ax.set(xlabel="Mean-normalized workload", ylabel="Cumulative probability", title="Workload representation distribution")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.18, wspace=0.3)
    _save_all(fig, "Fig_2_workload_validation")


def figure_3() -> None:
    data = pd.read_csv(TABLE_DIR / "v_d_binding_capacity_summary.csv")
    hard = data.loc[data.memory_mode.eq("hard")]
    pivot = hard.pivot(index="memory_available_gb_config", columns="workload_capacity_fraction", values="quantized_oracle_gap_percent_mean").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(5.7, 4.55))
    _panel_label(ax, "a")
    image = ax.imshow(pivot.to_numpy(), cmap="YlGnBu", aspect="auto")
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color="white" if value > np.nanmedian(pivot.to_numpy()) else PALETTE["ink"])
    ax.set_xticks(range(pivot.shape[1]), [f"{100*x:.0f}%" for x in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]), [f"{x:g}" for x in pivot.index])
    ax.set(xlabel="Workload capacity (% of base)", ylabel="Available memory (GB)", title="WA-MCBR gap to the quantized QDP optimum")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Projected quantized-objective gap (%)")
    fig.subplots_adjust(left=0.16, right=0.89, top=0.88, bottom=0.14)
    _save_all(fig, "Fig_3_binding_gap_heatmap")


def figure_4() -> None:
    data = pd.read_csv(TABLE_DIR / "v_d_binding_capacity_summary.csv")
    focus = data.loc[data.workload_capacity_fraction.eq(0.6)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.25))
    for label, ax in zip("abcd", axes.flat):
        _panel_label(ax, label)
    for mode in ["hard", "soft", "none"]:
        part = focus.loc[focus.memory_mode.eq(mode)].sort_values("memory_available_gb_config")
        axes[0, 0].plot(part.memory_available_gb_config, part.public_objective_J_mean, marker="o", ms=3.5, label=mode.capitalize(), color=MODE_COLORS[mode])
        axes[0, 1].plot(part.memory_available_gb_config, 100 * part.true_memory_violation_mean, marker="o", ms=3.5, label=mode.capitalize(), color=MODE_COLORS[mode])
    axes[0, 0].set(title="Common barrier-free objective", xlabel="Available memory (GB)", ylabel="Public objective J")
    axes[0, 1].set(title="True memory violations", xlabel="Available memory (GB)", ylabel="Violation rate (%)")
    axes[0, 0].legend()

    lowest = data.loc[data.workload_capacity_fraction.eq(0.2) & data.memory_available_gb_config.eq(1.5)].set_index("memory_mode")
    modes = ["hard", "soft", "none"]
    x = np.arange(3)
    width = 0.34
    axes[1, 0].bar(x - width / 2, [100 * lowest.loc[m, "workload_utilization_mean"] for m in modes], width, label="Workload", color=PALETTE["teal"])
    axes[1, 0].bar(x + width / 2, [100 * lowest.loc[m, "memory_utilization_mean"] for m in modes], width, label="Memory", color=PALETTE["coral"])
    axes[1, 0].axhline(100, color=PALETTE["ink"], ls="--", lw=0.9)
    axes[1, 0].set_xticks(x, [m.capitalize() for m in modes])
    axes[1, 0].set(title="Utilization under the tightest joint limit", ylabel="Utilization (%)")
    axes[1, 0].legend(ncol=2)

    hard = data.loc[data.memory_mode.eq("hard")]
    reasons = ["rejections_workload_mean", "rejections_memory_mean", "rejections_cost_mean"]
    reason_labels = ["Workload", "Memory", "Cost"]
    bottom = np.zeros(5)
    memory_values = sorted(hard.memory_available_gb_config.unique())
    for reason, reason_label, color in zip(reasons, reason_labels, [PALETTE["teal"], PALETTE["coral"], PALETTE["gray"]]):
        values = hard.groupby("memory_available_gb_config")[reason].mean().reindex(memory_values).to_numpy()
        axes[1, 1].bar(np.arange(5), values, bottom=bottom, label=reason_label, color=color)
        bottom += values
    axes[1, 1].set_xticks(np.arange(5), [f"{v:g}" for v in memory_values])
    axes[1, 1].set(title="Hard-model rejection reasons", xlabel="Available memory (GB)", ylabel="Mean rejected local tasks")
    axes[1, 1].legend(ncol=3, loc="upper right")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.1, hspace=0.42, wspace=0.28)
    _save_all(fig, "Fig_4_memory_models")


def figure_5() -> None:
    sensitivity = pd.read_csv(TABLE_DIR / "v_e_wireless_sensitivity_summary.csv")
    layers = pd.read_csv(TABLE_DIR / "v_e_distance_layer_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.45))
    for label, ax in zip("abc", axes):
        _panel_label(ax, label)

    for factor, label, color in [
        ("bandwidth_mhz", "Bandwidth", PALETTE["blue"]),
        ("image_payload_mb", "Image payload", PALETTE["coral"]),
    ]:
        part = sensitivity.loc[sensitivity.factor.eq(factor)].sort_values("value")
        x = np.arange(len(part))
        base = float(part.public_objective_J_mean.iloc[1])
        axes[0].plot(x, 100 * (part.public_objective_J_mean / base - 1), marker="o", ms=3.5, label=label, color=color)
    axes[0].axhline(0, color=PALETTE["gray"], lw=0.9)
    axes[0].set_xticks([0, 1, 2], ["Low", "Base", "High"])
    axes[0].set(title="Bandwidth and payload sensitivity", ylabel="Change in objective J (%)", xlabel="Parameter level")
    axes[0].legend()

    baseline = layers.loc[layers.factor.eq("bandwidth_mhz") & layers.value.eq(20.0)].set_index("distance_layer")
    ordered = ["near", "middle", "far"]
    axes[1].bar(np.arange(3), [100 * baseline.loc[layer, "offload_rate_mean"] for layer in ordered], color=[PALETTE["teal"], PALETTE["gold"], PALETTE["coral"]])
    axes[1].set_xticks(np.arange(3), [x.capitalize() for x in ordered])
    axes[1].set(title="Distance-stratified offloading", ylabel="Offload rate (%)", xlabel="User distance layer")

    labels = {
        "bandwidth_mhz": "BW",
        "image_payload_mb": "Payload",
        "uplink_power_w": "UL power",
        "downlink_power_w": "DL power",
        "path_loss_exponent": "Path loss",
    }
    for factor, label in labels.items():
        part = sensitivity.loc[sensitivity.factor.eq(factor)].sort_values("value")
        axes[2].plot(np.arange(3), part.jain_service_quality_mean, marker="o", ms=3.0, label=label)
    axes[2].set_xticks([0, 1, 2], ["Low", "Base", "High"])
    axes[2].set_ylim(0.79, 0.86)
    axes[2].set(title="Service-quality fairness", ylabel="Jain fairness index", xlabel="Parameter level")
    axes[2].legend(ncol=2)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.85, bottom=0.2, wspace=0.36)
    _save_all(fig, "Fig_5_wireless_fairness")


def figure_6() -> None:
    trace = pd.read_csv(RUN_DIR / "representative_convergence_trace.csv")
    scale = pd.read_csv(TABLE_DIR / "v_g_scale_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.15))
    for label, ax in zip("abcd", axes.flat):
        _panel_label(ax, label)

    axes[0, 0].plot(trace.iteration, trace.public_objective_J, color=PALETTE["blue"])
    accepted = trace.loc[trace.accepted.astype(bool)]
    axes[0, 0].scatter(accepted.iteration, accepted.public_objective_J, s=9, color=PALETTE["coral"], zorder=3, label="Accepted flip")
    axes[0, 0].set(title="Representative finite descent", xlabel="Asynchronous update", ylabel="Public objective J")
    axes[0, 0].legend()

    algorithm_style = {
        "wa_mcbr": ("WA-MCBR", PALETTE["blue"]),
        "legacy_count_br": ("Legacy count BR", PALETTE["gray"]),
    }
    for algorithm, (label, color) in algorithm_style.items():
        part = scale.loc[scale.algorithm.eq(algorithm)].sort_values("n_nodes")
        axes[0, 1].plot(part.n_nodes, part.updates_mean, marker="o", ms=3.5, color=color, label=label)
        axes[1, 0].plot(part.n_nodes, 1000 * part.runtime_seconds_mean, marker="o", ms=3.5, color=color, label=label)
    axes[0, 1].set(title="Update count with scale", xlabel="Number of tasks N", ylabel="Updates")
    axes[0, 1].legend()
    axes[1, 0].set(title="Online algorithm runtime", xlabel="Number of tasks N", ylabel="Runtime (ms)")

    wa = scale.loc[scale.algorithm.eq("wa_mcbr")].sort_values("n_nodes")
    axes[1, 1].plot(wa.n_nodes, wa.signaling_total_bytes_mean / 1024, marker="o", ms=3.5, color=PALETTE["teal"])
    axes[1, 1].fill_between(
        wa.n_nodes,
        wa.signaling_total_bytes_ci95_low / 1024,
        wa.signaling_total_bytes_ci95_high / 1024,
        color=PALETTE["teal"],
        alpha=0.18,
        linewidth=0,
    )
    axes[1, 1].set(title="Controller signaling", xlabel="Number of tasks N", ylabel="Total control traffic (KiB)")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.94, bottom=0.11, hspace=0.42, wspace=0.3)
    _save_all(fig, "Fig_6_convergence_runtime_signaling")


def _figure_contracts() -> pd.DataFrame:
    rows = [
        (1, "schematic-led composite", "A single-cell controller couples offloading count, workload, and memory under hard limits.", "model equations and configuration", "No deployment or optimality claim"),
        (2, "quantitative grid", "Observed workload contains task-level information, while DeepSeek remains a heuristic profiler below supervised baselines.", "profiler_predictions.csv; task_profiles.csv; Table II", "Do not claim DeepSeek superiority"),
        (3, "quantitative grid", "The projected WA-MCBR quantized gap varies across jointly binding workload and memory limits.", "v_d_binding_capacity_summary.csv", "Gap is to the quantized QDP optimum after projection"),
        (4, "quantitative grid", "Hard constraints preserve feasibility while soft and unconstrained policies trade feasibility against the common business objective.", "v_d_binding_capacity_summary.csv", "Barrier penalty is not included in the common objective"),
        (5, "quantitative grid", "Wireless bandwidth, payload, distance, and path loss alter offloading value and service-quality fairness.", "v_e_wireless_sensitivity_summary.csv; v_e_distance_layer_summary.csv", "Wireless inputs are simulated, not field measurements"),
        (6, "quantitative grid", "WA-MCBR exhibits finite descent with scale-dependent online runtime and signaling.", "representative_convergence_trace.csv; v_g_scale_summary.csv", "Convergence is single-flip local, not global"),
    ]
    return pd.DataFrame(rows, columns=["figure", "archetype", "core_conclusion", "source_data", "reviewer_risk_boundary"])


def _pixel_audit() -> pd.DataFrame:
    rows = []
    for directory, suffix in [("PNG", ".png"), ("TIFF", ".tiff")]:
        for path in sorted((FIGURE_ROOT / directory).glob(f"*{suffix}")):
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"))
                rows.append(
                    {
                        "format": directory,
                        "file": path.name,
                        "width_px": image.width,
                        "height_px": image.height,
                        "pixel_std": float(array.std()),
                        "nonblank_pass": bool(float(array.std()) > 5.0),
                        "resolution_pass": bool(image.width >= 1800 and image.height >= 1000),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    _configure_style()
    write_csv(AUDIT_DIR / "figure_contracts.csv", _figure_contracts())
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    figure_6()
    write_csv(AUDIT_DIR / "figure_pixel_audit.csv", _pixel_audit())
    print("Generated Fig. 1-6 in PNG, PDF, SVG, and TIFF formats")


if __name__ == "__main__":
    main()
