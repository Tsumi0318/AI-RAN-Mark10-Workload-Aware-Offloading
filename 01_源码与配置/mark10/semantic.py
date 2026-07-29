from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from mark10.data_pipeline import PROFILE_DIR, build_task_pools, load_raw_requests, persist_task_pools
from mark10.io_utils import ROOT, load_config, sha256_file, write_csv, write_json


AUDIT_DIR = ROOT / "06_审计与复现"


@dataclass(frozen=True)
class SemanticProfile:
    compute_multiplier: float
    memory_multiplier: float
    semantic_class: str
    warning: str


class SemanticClient(Protocol):
    def evaluate(self, intent: dict[str, Any]) -> dict[str, Any]: ...


def build_intent(row: pd.Series) -> dict[str, Any]:
    return {
        "task_uid": str(row["task_uid"]),
        "task_type": str(row["predict_type"]),
        "prompt_length_chars": int(round(float(row["prompt_length"]))),
        "negative_prompt_length_chars": int(round(float(row["negative_prompt_length"]))),
        "steps": int(round(float(row["num_inference_steps"]))),
        "num_images": int(round(float(row["num_images_per_prompt"]))),
        "lora_count": int(round(float(row["num_lora"]))),
    }


def parse_profile(text: str, minimum: float = 0.25, maximum: float = 4.0) -> SemanticProfile:
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError("response is not strict JSON") from exc
    required = {"compute_multiplier", "memory_multiplier", "semantic_class", "warning"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"missing fields: {missing}")
    values: dict[str, float] = {}
    for field in ["compute_multiplier", "memory_multiplier"]:
        try:
            value = float(payload[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} is not numeric") from exc
        if not minimum <= value <= maximum:
            raise ValueError(f"{field} outside [{minimum}, {maximum}]")
        values[field] = value
    semantic_class = str(payload["semantic_class"]).strip()
    warning = str(payload["warning"]).strip()
    if not semantic_class:
        raise ValueError("semantic_class must be nonempty")
    if not warning:
        warning = "none_reported"
    return SemanticProfile(values["compute_multiplier"], values["memory_multiplier"], semantic_class, warning)


class DeepSeekClient:
    def __init__(self, config: dict[str, Any]):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for fresh profiling")
        from openai import OpenAI

        self.config = config
        self.model = str(config["deepseek_model"])
        self.client = OpenAI(
            api_key=api_key,
            base_url=str(config["deepseek_base_url"]),
            timeout=float(config["deepseek_timeout_seconds"]),
        )

    def evaluate(self, intent: dict[str, Any]) -> dict[str, Any]:
        system = (
            "You are an offline semantic resource profiler for a reproducible AI-RAN simulation. "
            "Use only the supplied request features. Do not choose local versus offload. "
            "Return one strict JSON object with exactly compute_multiplier, memory_multiplier, "
            "semantic_class, and warning. Multipliers are relative resource demands in [0.25,4.0], "
            "where 1.0 is a baseline task. Keep warning concise."
        )
        last_error = ""
        for attempt in range(int(self.config["deepseek_retries"]) + 1):
            started = time.perf_counter()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=float(self.config["deepseek_temperature"]),
                    max_tokens=int(self.config["deepseek_max_tokens"]),
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(intent, ensure_ascii=False, sort_keys=True)},
                    ],
                )
                content = response.choices[0].message.content or ""
                profile = parse_profile(
                    content,
                    float(self.config["semantic_multiplier_min"]),
                    float(self.config["semantic_multiplier_max"]),
                )
                usage = response.usage
                return {
                    **asdict(profile),
                    "requested_model": self.model,
                    "resolved_model": str(response.model),
                    "latency_ms": (time.perf_counter() - started) * 1000.0,
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
                    "attempts": attempt + 1,
                    "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                }
            except Exception as exc:  # API and schema errors share the bounded retry policy.
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= int(self.config["deepseek_retries"]):
                    break
                time.sleep(min(2.0**attempt, 8.0))
        raise RuntimeError(f"DeepSeek profile failed for {intent['task_uid']}: {last_error}")


def _evaluate_one(client: SemanticClient, row: pd.Series) -> dict[str, Any]:
    intent = build_intent(row)
    return {"task_uid": intent["task_uid"], **client.evaluate(intent)}


def run_fresh_profiles(
    tasks: pd.DataFrame,
    client: SemanticClient,
    config: dict[str, Any],
    checkpoint_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if tasks.task_uid.nunique() != len(tasks):
        raise ValueError("Task identifiers must be unique")
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    workers = int(config["deepseek_workers"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_evaluate_one, client, row): str(row.task_uid) for _, row in tasks.iterrows()}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            rows.append(future.result())
            if checkpoint_path is not None:
                checkpoint = pd.DataFrame(rows).sort_values("task_uid").reset_index(drop=True)
                write_csv(checkpoint_path, checkpoint)
            if completed % 25 == 0:
                print(f"Fresh DeepSeek profiles: {completed}/{len(tasks)}", flush=True)
    profiles = pd.DataFrame(rows).sort_values("task_uid").reset_index(drop=True)
    expected = int(config["n_task_pools"]) * int(config["tasks_per_pool"])
    if len(profiles) != expected:
        raise RuntimeError(f"Expected {expected} profiles, received {len(profiles)}")
    manifest = {
        "run_id": datetime.now(timezone.utc).strftime("mark10_%Y%m%dT%H%M%SZ"),
        "selected_tasks": expected,
        "successful_profiles": len(profiles),
        "fresh_api_calls": len(profiles),
        "cache_hits": 0,
        "failed_profiles": 0,
        "temperature": float(config["deepseek_temperature"]),
        "requested_model": str(config["deepseek_model"]),
        "resolved_models": sorted(profiles.resolved_model.unique().tolist()),
        "total_tokens": int(profiles.total_tokens.sum()),
        "mean_latency_ms": float(profiles.latency_ms.mean()),
        "p95_latency_ms": float(profiles.latency_ms.quantile(0.95)),
        "elapsed_seconds": time.perf_counter() - started,
        "observed_execution_time_sent_to_llm": False,
        "api_key_saved": False,
        "role": "offline replaceable workload and simulated-memory profiler",
    }
    return profiles, manifest


def prepare_inputs() -> pd.DataFrame:
    config = load_config()
    pools = build_task_pools(load_raw_requests(), config)
    persist_task_pools(pools)
    tasks = pd.concat(pools, ignore_index=True)
    intents = pd.DataFrame([build_intent(row) for _, row in tasks.iterrows()])
    write_csv(PROFILE_DIR / "semantic_intents.csv", intents)
    return tasks


def run_fresh_stage() -> None:
    config = load_config()
    tasks = prepare_inputs()
    checkpoint_path = AUDIT_DIR / "deepseek_profiles_checkpoint.csv"
    profiles, manifest = run_fresh_profiles(
        tasks,
        DeepSeekClient(config),
        config,
        checkpoint_path=checkpoint_path,
    )
    output = PROFILE_DIR / "deepseek_resource_profiles.csv"
    write_csv(output, profiles)
    manifest["output_file"] = str(output.relative_to(ROOT))
    manifest["output_sha256"] = sha256_file(output)
    manifest["intent_file"] = str((PROFILE_DIR / "semantic_intents.csv").relative_to(ROOT))
    manifest["intent_sha256"] = sha256_file(PROFILE_DIR / "semantic_intents.csv")
    write_json(AUDIT_DIR / "deepseek_generation_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", required=True)
    parser.parse_args()
    run_fresh_stage()


if __name__ == "__main__":
    main()
