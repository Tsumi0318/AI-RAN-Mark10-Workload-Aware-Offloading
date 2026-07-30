from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mark10.summarize import (
    TABLE_III_COLUMNS,
    _symbol_details_table,
    build_main_profiler_table,
    build_main_symbols_table,
    build_main_comparison_table,
    choose_profiler_detail_source,
    format_table_iii_latex,
    jain_index,
    select_overall_profiler_rows,
    summarize_with_ci,
)
from mark10 import table_rendering


def test_table_i_contains_exact_pdf_main_notation_contract() -> None:
    table = build_main_symbols_table()

    assert table.columns.tolist() == ["Symbol", "Definition"]
    assert table["Symbol"].tolist() == [
        "x_i",
        "q_i",
        "v_i",
        "K(x)",
        "W(x)",
        "V(x)",
        "C_w",
        "V_avail",
        "B_total",
        "J(x)",
    ]


def test_table_ii_contains_only_compact_held_out_metrics() -> None:
    source = pd.DataFrame(
        {
            "model": ["count", "count", "deepseek", "linear", "tree"],
            "subset": [
                "test_pool_01",
                "all_out_of_pool",
                "all_out_of_pool",
                "all_out_of_pool",
                "all_out_of_pool",
            ],
            "n": [100, 500, 500, 500, 500],
            "constant_prediction": [True, True, False, False, False],
            "mae": [0.4, 0.47, 0.48, 0.37, 0.33],
            "rmse": [0.6, 0.69, 0.67, 0.55, 0.49],
            "r2": [0.0, 0.0, 0.04, 0.36, 0.50],
            "spearman": [0.0, 0.0, 0.23, 0.47, 0.55],
        }
    )

    table = build_main_profiler_table(source)

    assert table.columns.tolist() == [
        "Model",
        "MAE down",
        "RMSE down",
        "R2 up",
        "Spearman up",
    ]
    assert table["Model"].tolist() == [
        "Count",
        "LLM profiler (DeepSeek)",
        "Linear",
        "Tree",
    ]
    assert table["MAE down"].tolist() == pytest.approx([0.47, 0.48, 0.37, 0.33])


def test_table_ii_rendering_does_not_apply_best_or_second_best_emphasis(
    tmp_path,
    monkeypatch,
) -> None:
    table = pd.DataFrame(
        {
            "Model": ["Count", "DeepSeek", "Linear", "Tree"],
            "MAE down": [0.47, 0.48, 0.37, 0.33],
            "RMSE down": [0.69, 0.67, 0.55, 0.49],
            "R2 up": [0.00, 0.04, 0.36, 0.50],
            "Spearman up": [0.00, 0.23, 0.47, 0.55],
        }
    )
    source = tmp_path / "table_ii.csv"
    table.to_csv(source, index=False)
    captured: dict = {}

    def capture_draw_table(*args, **kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(table_rendering, "_draw_table", capture_draw_table)

    table_rendering.render_table_ii(source, tmp_path / "table_ii.png")

    assert captured.get("ranked_cells") is None
    assert "Bold" not in captured["note"]
    assert "underline" not in captured["note"]
    assert "Prediction target: mean-normalized Data workload" in captured["note"]


def test_detailed_symbols_follow_pdf_notation() -> None:
    table = _symbol_details_table()
    symbols = set(table["symbol"])

    assert {"x_i", "K(x)", "W(x)", "V(x)", "C_w", "V_avail", "D_q(W)", "J(x)"} <= symbols
    assert {"s_i", "K", "W", "V", "W_max", "V_max", "D_comp(W)", "J(s)"}.isdisjoint(symbols)


def test_profiler_detail_source_does_not_replace_pool_rows_with_aggregate_only() -> None:
    aggregate_only = pd.DataFrame(
        {"model": ["count"], "subset": ["all_out_of_pool"]}
    )
    detailed = pd.DataFrame(
        {
            "model": ["count", "count"],
            "subset": ["test_pool_01", "all_out_of_pool"],
        }
    )

    source, should_persist = choose_profiler_detail_source(aggregate_only, detailed)

    assert source.equals(detailed)
    assert should_persist is False


def test_profiler_detail_source_accepts_fresh_per_pool_metrics() -> None:
    fresh = pd.DataFrame(
        {
            "model": ["count", "count"],
            "subset": ["test_pool_01", "all_out_of_pool"],
        }
    )
    stale = pd.DataFrame(
        {"model": ["count"], "subset": ["all_out_of_pool"]}
    )

    source, should_persist = choose_profiler_detail_source(fresh, stale)

    assert source.equals(fresh)
    assert should_persist is True


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
