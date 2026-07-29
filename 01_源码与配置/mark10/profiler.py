from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mark10.data_pipeline import PROFILE_DIR
from mark10.io_utils import ROOT, load_config, write_csv


RUN_DIR = ROOT / "03_逐运行结果"
TABLE_DIR = ROOT / "04_汇总表格"
MODEL_FEATURES = [
    "prompt_length",
    "negative_prompt_length",
    "num_inference_steps",
    "num_images_per_prompt",
    "num_lora",
    "predict_type",
]
NUMERIC_FEATURES = MODEL_FEATURES[:-1]
CATEGORICAL_FEATURES = ["predict_type"]


def normalize_mean_one(values: np.ndarray | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("Workload values must have a finite positive mean")
    return array / mean


def _load_pools() -> pd.DataFrame:
    paths = sorted(PROFILE_DIR.glob("task_pool_*_pre_llm.csv"))
    if len(paths) != 5:
        raise RuntimeError(f"Expected five task pools, found {len(paths)}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def assemble_task_profiles(config: dict[str, Any]) -> pd.DataFrame:
    tasks = _load_pools()
    semantics = pd.read_csv(PROFILE_DIR / "deepseek_resource_profiles.csv")
    frame = tasks.merge(semantics, on="task_uid", how="inner", validate="one_to_one")
    if len(frame) != 500:
        raise RuntimeError(f"Expected 500 merged task profiles, found {len(frame)}")
    output = []
    for _, pool in frame.groupby("task_pool_id", sort=True):
        part = pool.copy()
        part["q_count"] = np.ones(len(part), dtype=float)
        part["q_data"] = normalize_mean_one(part.exec_time_seconds)
        part["q_llm"] = normalize_mean_one(part.compute_multiplier)
        output.append(part)
    result = pd.concat(output, ignore_index=True).sort_values(["task_pool_id", "task_index"])
    result["vram_base_gb"] = float(config["vram_base_gb"])
    result["vram_requirement_gb_simulated"] = result.vram_base_gb * result.memory_multiplier
    result["vram_value_classification"] = "simulated_vbase_times_deepseek_multiplier"
    return result.reset_index(drop=True)


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)],
        sparse_threshold=0.0,
    )


def _models(seed: int) -> dict[str, Pipeline]:
    return {
        "linear": Pipeline([("features", _preprocessor()), ("model", LinearRegression())]),
        "tree": Pipeline(
            [
                ("features", _preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=4,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def cross_pool_predictions(tasks: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    pool_ids = sorted(int(value) for value in tasks.task_pool_id.unique())
    for test_pool_id in pool_ids:
        test = tasks.loc[tasks.task_pool_id.eq(test_pool_id)].copy()
        train = tasks.loc[~tasks.task_pool_id.eq(test_pool_id)].copy()
        train_ids = ",".join(str(value) for value in pool_ids if value != test_pool_id)
        predictions: dict[str, np.ndarray] = {
            "count": test.q_count.to_numpy(float),
            "deepseek": test.q_llm.to_numpy(float),
        }
        for name, model in _models(int(config["seed"]) + test_pool_id).items():
            model.fit(train[MODEL_FEATURES], train.q_data.to_numpy(float))
            raw = np.maximum(model.predict(test[MODEL_FEATURES]), 0.05)
            predictions[name] = normalize_mean_one(raw)
        for model_name, values in predictions.items():
            rows.append(
                pd.DataFrame(
                    {
                        "task_uid": test.task_uid.to_numpy(),
                        "test_pool_id": test_pool_id,
                        "train_pool_ids": train_ids,
                        "model": model_name,
                        "q_true_data": test.q_data.to_numpy(float),
                        "q_predicted": values,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def profiler_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(truth, dtype=float)
    y_pred = np.asarray(prediction, dtype=float)
    if np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        correlation = 0.0
    else:
        correlation = float(spearmanr(y_true, y_pred).statistic)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": correlation,
    }


def build_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, model_frame in predictions.groupby("model", sort=True):
        for pool_id, pool_frame in model_frame.groupby("test_pool_id", sort=True):
            rows.append(
                {
                    "model": model,
                    "subset": f"test_pool_{int(pool_id):02d}",
                    "n": len(pool_frame),
                    "constant_prediction": bool(pool_frame.q_predicted.nunique() == 1),
                    **profiler_metrics(pool_frame.q_true_data, pool_frame.q_predicted),
                }
            )
        rows.append(
            {
                "model": model,
                "subset": "all_out_of_pool",
                "n": len(model_frame),
                "constant_prediction": bool(model_frame.q_predicted.nunique() == 1),
                **profiler_metrics(model_frame.q_true_data, model_frame.q_predicted),
            }
        )
    return pd.DataFrame(rows)


def workload_distribution_summary(tasks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pool_id, pool in tasks.groupby("task_pool_id", sort=True):
        for representation in ["q_count", "q_data", "q_llm"]:
            values = pool[representation].to_numpy(float)
            rows.append(
                {
                    "task_pool_id": int(pool_id),
                    "representation": representation,
                    "n": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "min": float(np.min(values)),
                    "median": float(np.median(values)),
                    "p95": float(np.quantile(values, 0.95)),
                    "max": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    tasks = assemble_task_profiles(config)
    predictions = cross_pool_predictions(tasks, config)
    metrics = build_metric_table(predictions)
    write_csv(PROFILE_DIR / "task_profiles.csv", tasks)
    write_csv(RUN_DIR / "profiler_predictions.csv", predictions)
    write_csv(TABLE_DIR / "table_ii_profiler_metrics.csv", metrics)
    write_csv(TABLE_DIR / "workload_distribution_summary.csv", workload_distribution_summary(tasks))


if __name__ == "__main__":
    main()
