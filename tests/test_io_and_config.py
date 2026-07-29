from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mark10.io_utils import load_config, sha256_file, write_csv, write_json


def test_config_defines_complete_protocol() -> None:
    config = load_config()
    assert config["n_task_pools"] == 5
    assert config["tasks_per_pool"] == 100
    assert config["wireless_seeds_per_pool"] == 10
    assert config["figures"] == [1, 2, 3, 4, 5, 6]
    assert config["workload_capacity_fractions"] == [0.2, 0.4, 0.6, 0.8, 1.0]
    assert config["memory_available_grid_gb"] == [1.5, 2.0, 3.0, 5.0, 13.0]


def test_write_csv_rejects_empty_frame(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty CSV"):
        write_csv(tmp_path / "empty.csv", pd.DataFrame())


def test_writers_create_parent_directories(tmp_path: Path) -> None:
    csv_path = tmp_path / "nested" / "rows.csv"
    json_path = tmp_path / "nested" / "value.json"
    write_csv(csv_path, pd.DataFrame([{"value": 1}]))
    write_json(json_path, {"value": 1})
    assert pd.read_csv(csv_path).to_dict("records") == [{"value": 1}]
    assert json_path.read_text(encoding="utf-8").strip().startswith("{")


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_text("Mark10\n", encoding="utf-8")
    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64
