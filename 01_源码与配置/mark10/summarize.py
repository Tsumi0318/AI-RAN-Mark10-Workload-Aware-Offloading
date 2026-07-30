from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t

from .io_utils import ROOT, load_config, write_csv


RUN_DIR = ROOT / "03_逐运行结果"
TABLE_DIR = ROOT / "04_汇总表格"

TABLE_III_COLUMNS = [
    "Scenario",
    "Algorithm",
    "J down",
    "Gap (%) down",
    "End-to-end delay (s) down",
    "Energy (J) down",
    "Violation rate (%) down",
    "Runtime (ms) down",
]
TABLE_III_SCENARIOS = {
    "abundant": "Resource-abundant",
    "moderate": "Moderately constrained",
    "highly_constrained": "Highly constrained",
}
TABLE_III_ALGORITHMS = {
    "qdp_oracle": "QDP-Oracle",
    "wa_mcbr": "WA-MCBR",
    "wa_mcbr_swap": "WA-MCBR-Swap",
    "capacity_greedy": "Capacity-aware Greedy",
    "lagrangian_relaxation": "Lagrangian/Primal-dual",
    "legacy_count_br": "Legacy",
    "random_feasible": "Random",
    "all_local": "All-local",
    "all_offload": "All-offload",
}


def jain_index(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Jain fairness requires finite nonnegative values")
    denominator = array.size * float(np.square(array).sum())
    return float(array.sum() ** 2 / denominator) if denominator > 0 else 1.0


def _independent_instance_count(frame: pd.DataFrame) -> int:
    columns = [name for name in ["task_pool_id", "wireless_index"] if name in frame.columns]
    if not columns:
        return len(frame)
    return int(frame[columns].drop_duplicates().shape[0])


def summarize_with_ci(
    frame: pd.DataFrame,
    keys: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    missing = set(keys).union(metrics) - set(frame.columns)
    if missing:
        raise KeyError(f"Missing summary columns: {sorted(missing)}")
    rows: list[dict[str, float | int | str]] = []
    grouped = frame.groupby(list(keys), dropna=False, sort=True)
    for group_value, part in grouped:
        values = group_value if isinstance(group_value, tuple) else (group_value,)
        row: dict[str, float | int | str] = dict(zip(keys, values))
        row["independent_instances"] = _independent_instance_count(part)
        for metric in metrics:
            data = pd.to_numeric(part[metric], errors="coerce").to_numpy(float)
            data = data[np.isfinite(data)]
            n = int(data.size)
            mean = float(data.mean()) if n else math.nan
            std = float(data.std(ddof=1)) if n > 1 else 0.0 if n == 1 else math.nan
            half_width = float(t.ppf(0.975, n - 1) * std / math.sqrt(n)) if n > 1 else 0.0
            row[f"{metric}_n"] = n
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = mean - half_width if n else math.nan
            row[f"{metric}_ci95_high"] = mean + half_width if n else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _symbols_table() -> pd.DataFrame:
    rows = [
        ("N", "number of edge tasks/nodes", "task", "experiment input"),
        ("s_i", "binary decision: 1 offload, 0 local", "binary", "decision variable"),
        ("K", "number of offloaded tasks", "task", "sum_i s_i"),
        ("q_i", "normalized workload of task i", "dimensionless", "profiler output"),
        ("W", "aggregate offloaded workload", "dimensionless", "sum_i s_i q_i"),
        ("W_max", "workload hard limit", "dimensionless", "simulation parameter"),
        ("v_i", "simulated memory demand of task i", "GB", "v_base times LLM multiplier"),
        ("V", "aggregate offloaded memory demand", "GB", "sum_i s_i v_i"),
        ("V_max", "available edge memory hard limit", "GB", "simulation parameter"),
        ("R_i^up", "uplink rate of task i", "bit/s", "wireless model"),
        ("R_i^down", "downlink rate of task i", "bit/s", "wireless model"),
        ("D_comp(W)", "workload-dependent edge congestion delay", "s", "queue proxy"),
        ("C_i^loc", "local execution cost", "normalized cost", "delay-energy aggregate"),
        ("C_i^off", "offloading cost", "normalized cost", "radio, execution, congestion"),
        ("J(s)", "barrier-free public business objective", "normalized cost", "evaluation target"),
        ("Delta_q", "workload quantization step", "dimensionless", "QDP parameter"),
        ("Delta_v", "memory quantization step", "GB", "QDP parameter"),
    ]
    return pd.DataFrame(rows, columns=["symbol", "definition", "unit", "provenance"])


def _setup_table(config: dict) -> pd.DataFrame:
    rows = [
        ("tasks", "task_pools", config["n_task_pools"], "pool", "five disjoint source dates"),
        ("tasks", "tasks_per_pool", config["tasks_per_pool"], "task", "GenTD26 successful requests"),
        ("wireless", "wireless_instances_per_pool", config["wireless_seeds_per_pool"], "instance", "independent position and shadowing seeds"),
        ("wireless", "bandwidth", config["bandwidth_hz"] / 1e6, "MHz", "simulated"),
        ("wireless", "uplink_power", config["uplink_power_w"], "W", "simulated"),
        ("wireless", "downlink_power", config["downlink_power_w"], "W", "simulated"),
        ("wireless", "path_loss_exponent", config["path_loss_exponent"], "dimensionless", "simulated"),
        ("resources", "base_workload_capacity", config["base_workload_capacity"], "dimensionless", "simulated"),
        ("resources", "memory_grid", ",".join(map(str, config["memory_available_grid_gb"])), "GB", "simulated"),
        ("resources", "vram_base", config["vram_base_gb"], "GB/task", "simulated; scaled by DeepSeek multiplier"),
        ("oracle", "workload_quantization", config["oracle_delta_q"], "dimensionless", "upward rounding"),
        ("oracle", "memory_quantization", config["oracle_delta_v_gb"], "GB", "upward rounding"),
        ("statistics", "independent_main_instances", config["n_task_pools"] * config["wireless_seeds_per_pool"], "instance", "algorithm order is not an independent instance"),
    ]
    return pd.DataFrame(rows, columns=["category", "parameter", "value", "unit", "note"])


def select_overall_profiler_rows(source: pd.DataFrame) -> pd.DataFrame:
    overall = source.loc[source.subset.eq("all_out_of_pool")].copy()
    if overall.empty:
        raise ValueError("Profiler metrics contain no all_out_of_pool rows")
    return overall


def _overall_profiler_table() -> pd.DataFrame:
    source = pd.read_csv(TABLE_DIR / "table_ii_profiler_metrics.csv")
    write_csv(TABLE_DIR / "v_b_profiler_metrics_by_pool.csv", source)
    overall = select_overall_profiler_rows(source)
    overall.insert(3, "target_unit", "mean-normalized workload")
    return overall


def _finite_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.mean()) if numeric.size else math.nan


def build_main_comparison_table(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "resource_scenario",
        "algorithm",
        "strategy_feasible_estimated",
        "public_objective_J",
        "quantized_oracle_gap_percent",
        "mean_end_to_end_delay_seconds",
        "total_device_energy_j",
        "true_workload_violation",
        "true_memory_violation",
        "runtime_seconds",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing Table III columns: {sorted(missing)}")

    source = frame.loc[
        frame["resource_scenario"].isin(TABLE_III_SCENARIOS)
        & frame["algorithm"].isin(TABLE_III_ALGORITHMS)
    ].copy()
    source["joint_violation"] = source[
        ["true_workload_violation", "true_memory_violation"]
    ].max(axis=1)

    rows: list[dict[str, float | str]] = []
    for (scenario, algorithm), part in source.groupby(
        ["resource_scenario", "algorithm"], sort=False
    ):
        all_infeasible = pd.to_numeric(
            part["strategy_feasible_estimated"], errors="coerce"
        ).eq(0).all()
        rows.append(
            {
                "Scenario": TABLE_III_SCENARIOS[scenario],
                "Algorithm": TABLE_III_ALGORITHMS[algorithm],
                "J down": _finite_mean(part["public_objective_J"]),
                "Gap (%) down": math.nan
                if all_infeasible
                else _finite_mean(part["quantized_oracle_gap_percent"]),
                "End-to-end delay (s) down": _finite_mean(
                    part["mean_end_to_end_delay_seconds"]
                ),
                "Energy (J) down": _finite_mean(part["total_device_energy_j"]),
                "Violation rate (%) down": 100.0
                * _finite_mean(part["joint_violation"]),
                "Runtime (ms) down": 1000.0 * _finite_mean(part["runtime_seconds"]),
            }
        )

    result = pd.DataFrame(rows, columns=TABLE_III_COLUMNS)
    scenario_order = {name: index for index, name in enumerate(TABLE_III_SCENARIOS.values())}
    algorithm_order = {name: index for index, name in enumerate(TABLE_III_ALGORITHMS.values())}
    result["_scenario_order"] = result["Scenario"].map(scenario_order)
    result["_algorithm_order"] = result["Algorithm"].map(algorithm_order)
    return (
        result.sort_values(["_scenario_order", "_algorithm_order"])
        .drop(columns=["_scenario_order", "_algorithm_order"])
        .reset_index(drop=True)
    )


def _table_iii_rank_classes(table: pd.DataFrame) -> dict[tuple[int, str], str]:
    classes: dict[tuple[int, str], str] = {}
    metric_columns = TABLE_III_COLUMNS[2:]
    violation_column = "Violation rate (%) down"
    for _, part in table.groupby("Scenario", sort=False):
        for column in metric_columns:
            numeric = pd.to_numeric(part[column], errors="coerce")
            eligible = numeric.notna()
            if column != violation_column:
                eligible &= pd.to_numeric(
                    part[violation_column], errors="coerce"
                ).eq(0)
            distinct = sorted(numeric.loc[eligible].unique())
            if not distinct:
                continue
            best = distinct[0]
            best_indices = part.index[eligible & numeric.eq(best)]
            for index in best_indices:
                classes[(int(index), column)] = "best"
            if len(best_indices) == 1 and len(distinct) > 1:
                second = distinct[1]
                for index in part.index[eligible & numeric.eq(second)]:
                    classes[(int(index), column)] = "second"
    return classes


def format_table_iii_latex(table: pd.DataFrame) -> str:
    missing = set(TABLE_III_COLUMNS) - set(table.columns)
    if missing:
        raise KeyError(f"Missing formatted Table III columns: {sorted(missing)}")
    ranks = _table_iii_rank_classes(table)
    formats = {
        "J down": ".3f",
        "Gap (%) down": ".3f",
        "End-to-end delay (s) down": ".2f",
        "Energy (J) down": ".1f",
        "Violation rate (%) down": ".1f",
        "Runtime (ms) down": ".2f",
    }

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Overall algorithm comparison across three resource regimes.}",
        r"\label{tab:overall_algorithm_comparison}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Scenario & Algorithm & $J\downarrow$ & Gap (\%)$\downarrow$ & E2E delay (s)$\downarrow$ & Energy (J)$\downarrow$ & Violation (\%)$\downarrow$ & Runtime (ms)$\downarrow$ \\",
        r"\midrule",
    ]
    previous_scenario: str | None = None
    for index, row in table.iterrows():
        scenario = str(row["Scenario"])
        if previous_scenario is not None and scenario != previous_scenario:
            lines.append(r"\midrule")
        scenario_cell = scenario if scenario != previous_scenario else ""
        cells = [scenario_cell, str(row["Algorithm"])]
        for column in TABLE_III_COLUMNS[2:]:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if not np.isfinite(value):
                cell = "--"
            else:
                cell = format(float(value), formats[column])
                rank_class = ranks.get((int(index), column))
                if rank_class == "best":
                    cell = rf"\textbf{{{cell}}}"
                elif rank_class == "second":
                    cell = rf"\underline{{{cell}}}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
        previous_scenario = scenario
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\begin{minipage}{\textwidth}\footnotesize Values are means over 50 independent task-pool/wireless instances. Bold and underlined values denote the best and second-best values, respectively; tied best values are all bold and no second-best is marked. Rankings exclude methods with nonzero hard-constraint violation; the violation column ranks all methods. QDP-Oracle is an offline benchmark. ``--'' denotes an undefined metric.\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _main_table() -> pd.DataFrame:
    return build_main_comparison_table(
        pd.read_csv(RUN_DIR / "main_algorithm_runs.csv")
    )


def build_all_tables() -> dict[str, pd.DataFrame]:
    config = load_config()
    outputs: dict[str, pd.DataFrame] = {
        "table_i_symbols.csv": _symbols_table(),
        "table_ii_profiler_metrics.csv": _overall_profiler_table(),
        "table_iii_algorithm_comparison.csv": _main_table(),
        "v_a_experimental_setup.csv": _setup_table(config),
    }

    binding = pd.read_csv(RUN_DIR / "binding_resource_runs.csv")
    binding_metrics = [
        "public_objective_J", "quantized_oracle_gap_percent", "offload_rate",
        "workload_utilization", "memory_utilization", "true_workload_violation",
        "true_memory_violation", "barrier_penalty", "rejections_workload",
        "rejections_memory", "rejections_cost",
    ]
    outputs["v_d_binding_capacity_summary.csv"] = summarize_with_ci(
        binding,
        ["workload_capacity_fraction", "memory_available_gb_config", "memory_mode"],
        binding_metrics,
    )
    outputs["v_d_memory_model_summary.csv"] = summarize_with_ci(
        binding, ["memory_mode"], binding_metrics
    )

    wireless = pd.read_csv(RUN_DIR / "wireless_sensitivity_runs.csv")
    outputs["v_e_wireless_sensitivity_summary.csv"] = summarize_with_ci(
        wireless,
        ["factor", "value"],
        ["public_objective_J", "offload_rate", "mean_end_to_end_delay_seconds", "jain_service_quality"],
    )
    distance = pd.read_csv(RUN_DIR / "wireless_distance_layers.csv")
    outputs["v_e_distance_layer_summary.csv"] = summarize_with_ci(
        distance,
        ["factor", "value", "distance_layer"],
        ["offload_rate", "mean_end_to_end_delay_seconds", "mean_service_quality"],
    )

    robustness = pd.read_csv(RUN_DIR / "profiling_error_robustness.csv")
    outputs["v_f_profile_error_summary.csv"] = summarize_with_ci(
        robustness,
        ["error_magnitude", "q_error_sign", "v_error_sign", "protection"],
        ["objective_degradation_percent", "strategy_change_rate", "true_workload_violation", "true_memory_violation"],
    )

    scale = pd.read_csv(RUN_DIR / "scale_runs.csv")
    outputs["v_g_scale_summary.csv"] = summarize_with_ci(
        scale,
        ["n_nodes", "algorithm"],
        ["public_objective_J", "updates", "runtime_seconds", "signaling_total_bytes"],
    )
    oracle = pd.read_csv(RUN_DIR / "scale_oracle_diagnostics.csv")
    outputs["v_g_qdp_complexity.csv"] = oracle.copy()
    outputs["v_h_failure_cases.csv"] = pd.read_csv(RUN_DIR / "failure_cases.csv")

    for filename, table in outputs.items():
        write_csv(TABLE_DIR / filename, table)
    (TABLE_DIR / "table_iii_algorithm_comparison.tex").write_text(
        format_table_iii_latex(outputs["table_iii_algorithm_comparison.csv"]),
        encoding="utf-8",
    )
    return outputs


def main() -> None:
    outputs = build_all_tables()
    for filename, table in outputs.items():
        print(f"{filename}: {len(table)} rows")
    print("table_iii_algorithm_comparison.tex: formatted double-column table")


if __name__ == "__main__":
    main()
