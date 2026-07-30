from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mark10.summarize import (
    TABLE_III_COLUMNS,
    build_main_comparison_table,
    format_table_iii_latex,
    jain_index,
    select_overall_profiler_rows,
    summarize_with_ci,
)


def test_jain_index_is_one_for_equal_values() -> None:
    assert jain_index(np.array([2.0, 2.0, 2.0])) == pytest.approx(1.0)


def test_jain_index_rejects_negative_or_empty_values() -> None:
    with pytest.raises(ValueError):
        jain_index(np.array([]))
    with pytest.raises(ValueError):
        jain_index(np.array([1.0, -1.0]))


def test_summary_counts_independent_instances_and_builds_ci() -> None:
    rows = []
    for pool_id in range(1, 6):
        for wireless_index in range(10):
            rows.append(
                {
                    "scenario": "moderate",
                    "algorithm": "wa_mcbr",
                    "task_pool_id": pool_id,
                    "wireless_index": wireless_index,
                    "objective_J": float(pool_id + wireless_index),
                }
            )
    frame = pd.DataFrame(rows)
    summary = summarize_with_ci(
        frame,
        ["scenario", "algorithm"],
        ["objective_J"],
    )
    assert summary.loc[0, "independent_instances"] == 50
    assert summary.loc[0, "objective_J_n"] == 50
    assert summary.loc[0, "objective_J_ci95_low"] < summary.loc[0, "objective_J_mean"]
    assert summary.loc[0, "objective_J_ci95_high"] > summary.loc[0, "objective_J_mean"]


def test_summary_excludes_nonfinite_metric_values_but_keeps_instance_count() -> None:
    frame = pd.DataFrame(
        {
            "scenario": ["x", "x", "x"],
            "task_pool_id": [1, 1, 1],
            "wireless_index": [0, 1, 2],
            "objective_J": [1.0, np.inf, 3.0],
        }
    )
    summary = summarize_with_ci(frame, ["scenario"], ["objective_J"])
    assert summary.loc[0, "independent_instances"] == 3
    assert summary.loc[0, "objective_J_n"] == 2
    assert summary.loc[0, "objective_J_mean"] == pytest.approx(2.0)


def test_profiler_table_uses_out_of_pool_aggregate_rows() -> None:
    source = pd.DataFrame(
        {
            "model": ["count", "count", "tree", "tree"],
            "subset": ["test_pool_01", "all_out_of_pool", "test_pool_01", "all_out_of_pool"],
            "n": [100, 500, 100, 500],
        }
    )
    result = select_overall_profiler_rows(source)
    assert result.model.tolist() == ["count", "tree"]
    assert result.n.tolist() == [500, 500]


def test_main_comparison_table_has_required_columns_and_joint_violation_rate() -> None:
    frame = pd.DataFrame(
        {
            "resource_scenario": ["abundant", "abundant"],
            "algorithm": ["wa_mcbr", "wa_mcbr"],
            "strategy_feasible_estimated": [1, 0],
            "public_objective_J": [10.0, 14.0],
            "quantized_oracle_gap_percent": [1.0, 3.0],
            "mean_end_to_end_delay_seconds": [2.0, 4.0],
            "total_device_energy_j": [20.0, 24.0],
            "true_workload_violation": [1, 0],
            "true_memory_violation": [1, 0],
            "runtime_seconds": [0.01, 0.03],
        }
    )

    result = build_main_comparison_table(frame)

    assert result.columns.tolist() == TABLE_III_COLUMNS
    assert result.loc[0, "J down"] == pytest.approx(12.0)
    assert result.loc[0, "Violation rate (%) down"] == pytest.approx(50.0)
    assert result.loc[0, "Runtime (ms) down"] == pytest.approx(20.0)


def test_table_iii_latex_marks_feasible_best_and_second_best() -> None:
    table = pd.DataFrame(
        [
            ["Resource-abundant", "QDP-Oracle", 10.0, 0.0, 3.0, 20.0, 0.0, 30.0],
            ["Resource-abundant", "WA-MCBR", 11.0, 1.0, 2.0, 21.0, 0.0, 20.0],
            ["Resource-abundant", "All-offload", np.nan, np.nan, np.nan, 1.0, 100.0, 0.0],
        ],
        columns=TABLE_III_COLUMNS,
    )

    latex = format_table_iii_latex(table)

    assert r"\textbf{10.000}" in latex
    assert r"\underline{11.000}" in latex
    assert r"\textbf{1.0}" not in latex
    assert r"\textbf{100.0}" not in latex
    assert r"\underline{100.0}" not in latex
    assert "--" in latex
