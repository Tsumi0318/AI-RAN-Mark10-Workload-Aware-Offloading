from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mark10.io_utils import RAW_DATA, ROOT, load_config, write_csv


PROFILE_DIR = ROOT / "02_任务池与画像"


def load_raw_requests(path: Path | None = None) -> pd.DataFrame:
    source = path or RAW_DATA / "lora_request_trace.csv"
    frame = pd.read_csv(source)
    frame["source_row"] = np.arange(len(frame), dtype=int)
    frame["gmt_create"] = pd.to_datetime(frame["gmt_create"], errors="coerce")
    return frame


def _usable_requests(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.loc[
        raw.predict_status.eq("SUCCEED")
        & pd.to_numeric(raw.exec_time_seconds, errors="coerce").gt(0)
        & raw.gmt_create.notna()
        & raw.predict_type.isin(["TXT_2_IMG", "IMG_2_IMG", "INPAINTING"])
    ].copy()
    numeric_defaults = {
        "prompt_length": None,
        "negative_prompt_length": 0.0,
        "num_images_per_prompt": 1.0,
        "num_inference_steps": None,
        "num_lora": 0.0,
    }
    for column, default in numeric_defaults.items():
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        fill = float(frame[column].median()) if default is None else default
        frame[column] = frame[column].fillna(fill)
    frame["source_date"] = frame.gmt_create.dt.strftime("%Y-%m-%d")
    return frame


def _eligible_dates(frame: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    minimums = config["task_type_minimums_per_pool"]
    counts = frame.groupby(["source_date", "predict_type"]).size().unstack(fill_value=0)
    eligible = []
    for date, row in counts.sort_index().iterrows():
        if all(int(row.get(task_type, 0)) >= int(required) for task_type, required in minimums.items()):
            eligible.append(str(date))
    if len(eligible) < int(config["n_task_pools"]):
        raise ValueError(f"Only {len(eligible)} dates satisfy task-type minimums")
    positions = np.linspace(0, len(eligible) - 1, int(config["n_task_pools"])).round().astype(int)
    selected = [eligible[index] for index in positions]
    if len(set(selected)) != len(selected):
        raise AssertionError("Date selection did not produce unique source dates")
    return selected


def _sample_one_pool(
    dated: pd.DataFrame,
    pool_index: int,
    config: dict[str, Any],
) -> pd.DataFrame:
    rng = np.random.default_rng(int(config["seed"]) + pool_index * 1009)
    selected_indices: list[int] = []
    for task_type, required in config["task_type_minimums_per_pool"].items():
        candidates = dated.index[dated.predict_type.eq(task_type)].to_numpy()
        selected_indices.extend(rng.choice(candidates, int(required), replace=False).tolist())
    remaining_count = int(config["tasks_per_pool"]) - len(selected_indices)
    remaining = dated.index[~dated.index.isin(selected_indices)].to_numpy()
    selected_indices.extend(rng.choice(remaining, remaining_count, replace=False).tolist())
    pool = dated.loc[selected_indices].copy()
    pool = pool.sort_values(["gmt_create", "source_row"]).reset_index(drop=True)
    pool["task_pool_id"] = pool_index
    pool["task_index"] = np.arange(len(pool), dtype=int)
    pool["task_uid"] = [f"pool{pool_index:02d}-task{index:03d}" for index in pool.task_index]
    return pool


def build_task_pools(raw: pd.DataFrame, config: dict[str, Any]) -> list[pd.DataFrame]:
    usable = _usable_requests(raw)
    dates = _eligible_dates(usable, config)
    pools = [
        _sample_one_pool(usable.loc[usable.source_date.eq(date)], index, config)
        for index, date in enumerate(dates, start=1)
    ]
    combined = pd.concat(pools, ignore_index=True)
    if combined.source_row.nunique() != int(config["n_task_pools"]) * int(config["tasks_per_pool"]):
        raise AssertionError("Task pools are not globally disjoint")
    return pools


def persist_task_pools(pools: list[pd.DataFrame]) -> None:
    for index, pool in enumerate(pools, start=1):
        write_csv(PROFILE_DIR / f"task_pool_{index:02d}_pre_llm.csv", pool)


def main() -> None:
    pools = build_task_pools(load_raw_requests(), load_config())
    persist_task_pools(pools)


if __name__ == "__main__":
    main()

