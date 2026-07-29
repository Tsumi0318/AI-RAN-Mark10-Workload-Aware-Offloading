from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "01_源码与配置" / "config.json"
RAW_DATA = ROOT / "00_原始数据" / "GenTD26"
AUDIT_DIR = ROOT / "06_审计与复现"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def build_source_manifest() -> pd.DataFrame:
    rows = []
    source_url = "https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI"
    for path in sorted(RAW_DATA.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "file": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "source": source_url,
                    "classification": "official_anonymized_production_trace",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="store_true")
    args = parser.parse_args()
    if args.manifest:
        write_csv(AUDIT_DIR / "source_data_manifest.csv", build_source_manifest())


if __name__ == "__main__":
    main()
