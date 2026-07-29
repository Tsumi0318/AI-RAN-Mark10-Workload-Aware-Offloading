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


def _main_table() -> pd.DataFrame:
    frame = pd.read_csv(RUN_DIR / "main_algorithm_runs.csv")
    metrics = [
        "public_objective_J",
        "quantized_oracle_gap_percent",
        "mean_end_to_end_delay_seconds",
        "total_device_energy_j",
        "offload_rate",
        "workload_utilization",
        "memory_utilization",
        "true_workload_violation",
        "true_memory_violation",
        "runtime_seconds",
        "updates",
        "signaling_total_bytes",
    ]
    summary = summarize_with_ci(frame, ["resource_scenario", "algorithm"], metrics)
    feasible = summary["true_workload_violation_mean"].eq(0) & summary["true_memory_violation_mean"].eq(0)
    summary["feasible_in_all_instances"] = feasible
    summary["objective_rank_within_scenario"] = np.nan
    for scenario, index in summary.loc[feasible].groupby("resource_scenario").groups.items():
        ranks = summary.loc[index, "public_objective_J_mean"].rank(method="min")
        summary.loc[index, "objective_rank_within_scenario"] = ranks
    return summary


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
    return outputs


def main() -> None:
    outputs = build_all_tables()
    for filename, table in outputs.items():
        print(f"{filename}: {len(table)} rows")


if __name__ == "__main__":
    main()
