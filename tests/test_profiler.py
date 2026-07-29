from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mark10.io_utils import load_config
from mark10.profiler import (
    MODEL_FEATURES,
    assemble_task_profiles,
    cross_pool_predictions,
    normalize_mean_one,
    profiler_metrics,
)


def test_normalize_mean_one_rejects_nonpositive_mean() -> None:
    with pytest.raises(ValueError, match="positive mean"):
        normalize_mean_one(np.array([0.0, 0.0]))


def test_assembled_workloads_have_mean_one_per_pool() -> None:
    tasks = assemble_task_profiles(load_config())
    assert len(tasks) == 500
    for _, pool in tasks.groupby("task_pool_id"):
        for column in ["q_count", "q_data", "q_llm"]:
            assert float(pool[column].mean()) == pytest.approx(1.0)
    assert np.allclose(tasks.vram_requirement_gb_simulated, tasks.vram_base_gb * tasks.memory_multiplier)


def test_predictive_features_do_not_include_execution_time() -> None:
    assert "exec_time_seconds" not in MODEL_FEATURES
    assert all("exec" not in feature.lower() for feature in MODEL_FEATURES)


def test_cross_pool_predictions_are_out_of_pool() -> None:
    predictions = cross_pool_predictions(assemble_task_profiles(load_config()), load_config())
    assert len(predictions) == 500 * 4
    for row in predictions[["test_pool_id", "train_pool_ids"]].drop_duplicates().itertuples(index=False):
        train_ids = {int(value) for value in row.train_pool_ids.split(",")}
        assert int(row.test_pool_id) not in train_ids
        assert len(train_ids) == 4


def test_profiler_metrics_are_finite() -> None:
    truth = np.array([0.5, 1.0, 1.5, 2.0])
    prediction = np.array([0.6, 0.9, 1.4, 2.1])
    metrics = profiler_metrics(truth, prediction)
    assert set(metrics) == {"mae", "rmse", "r2", "spearman"}
    assert all(np.isfinite(value) for value in metrics.values())


def test_constant_prediction_returns_zero_spearman_without_warning() -> None:
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        metrics = profiler_metrics(np.array([0.5, 1.0, 1.5]), np.ones(3))
    assert len(recorded) == 0
    assert metrics["spearman"] == 0.0
