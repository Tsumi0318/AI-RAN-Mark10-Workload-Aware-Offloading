from __future__ import annotations

import pandas as pd

from mark10.data_pipeline import build_task_pools, load_raw_requests
from mark10.io_utils import load_config


def test_task_pools_are_disjoint_and_complete() -> None:
    config = load_config()
    pools = build_task_pools(load_raw_requests(), config)
    assert len(pools) == 5
    assert [len(pool) for pool in pools] == [100] * 5
    selected = pd.concat(pools, ignore_index=True)
    assert selected.source_row.nunique() == 500
    assert selected.task_pool_id.nunique() == 5


def test_task_pools_are_date_separated() -> None:
    pools = build_task_pools(load_raw_requests(), load_config())
    date_sets = [set(pool.source_date) for pool in pools]
    for index, dates in enumerate(date_sets):
        for other in date_sets[index + 1 :]:
            assert dates.isdisjoint(other)


def test_each_pool_covers_all_available_task_types() -> None:
    pools = build_task_pools(load_raw_requests(), load_config())
    required = {"TXT_2_IMG", "IMG_2_IMG", "INPAINTING"}
    for pool in pools:
        assert required.issubset(set(pool.predict_type))


def test_task_pool_generation_is_deterministic() -> None:
    config = load_config()
    first = build_task_pools(load_raw_requests(), config)
    second = build_task_pools(load_raw_requests(), config)
    for left, right in zip(first, second, strict=True):
        assert left.source_row.tolist() == right.source_row.tolist()

