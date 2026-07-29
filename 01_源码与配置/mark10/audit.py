from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pandas as pd

from .io_utils import AUDIT_DIR, RAW_DATA, ROOT, sha256_file, write_csv, write_json


RUN_DIR = ROOT / "03_逐运行结果"
TABLE_DIR = ROOT / "04_汇总表格"
FIGURE_ROOT = ROOT / "05_论文图表"
EXPECTED_FIGURES = [
    "Fig_1_system_model",
    "Fig_2_workload_validation",
    "Fig_3_binding_gap_heatmap",
    "Fig_4_memory_models",
    "Fig_5_wireless_fairness",
    "Fig_6_convergence_runtime_signaling",
]
FIGURE_FORMATS = {"PNG": ".png", "PDF": ".pdf", "SVG": ".svg", "TIFF": ".tiff"}


def _check(rows: list[dict[str, str]], requirement: str, passed: bool, detail: str) -> None:
    rows.append({"requirement": requirement, "status": "PASS" if passed else "FAIL", "detail": detail})


def _same_source_hashes() -> tuple[bool, str]:
    manifest = pd.read_csv(AUDIT_DIR / "source_data_manifest.csv")
    actual = {path.name: sha256_file(path) for path in RAW_DATA.iterdir() if path.is_file()}
    expected = dict(zip(manifest.file, manifest.sha256))
    return actual == expected, f"{len(actual)} source files checked against SHA-256 manifest"


def build_audit() -> dict:
    rows: list[dict[str, str]] = []
    manifest = json.loads((AUDIT_DIR / "deepseek_generation_manifest.json").read_text(encoding="utf-8"))
    _check(
        rows,
        "fresh_deepseek_profiles",
        manifest["selected_tasks"] == 500
        and manifest["successful_profiles"] == 500
        and manifest["fresh_api_calls"] == 500
        and manifest["failed_profiles"] == 0,
        "500 selected, successful, and fresh API profiles; 0 failures",
    )
    _check(rows, "deepseek_cache_is_zero", manifest["cache_hits"] == 0, "cache_hits=0")
    _check(rows, "execution_time_not_sent_to_llm", not manifest["observed_execution_time_sent_to_llm"], "manifest=false")
    _check(rows, "api_key_not_saved", not manifest["api_key_saved"], "manifest=false")
    hashes_ok, hash_detail = _same_source_hashes()
    _check(rows, "source_data_sha256", hashes_ok, hash_detail)

    tasks = pd.read_csv(ROOT / "02_任务池与画像" / "task_profiles.csv")
    pool_sizes = tasks.groupby("task_pool_id").size().to_dict()
    _check(
        rows,
        "five_disjoint_task_pools",
        len(tasks) == 500 and tasks.source_row.nunique() == 500 and pool_sizes == {1: 100, 2: 100, 3: 100, 4: 100, 5: 100},
        f"rows={len(tasks)}, unique_source_rows={tasks.source_row.nunique()}, pool_sizes={pool_sizes}",
    )

    main = pd.read_csv(RUN_DIR / "main_algorithm_runs.csv")
    main_instances = main[["task_pool_id", "wireless_index"]].drop_duplicates()
    wa = main.loc[main.algorithm.eq("wa_mcbr")]
    _check(rows, "main_matrix_coverage", len(main) == 1350 and len(main_instances) == 50, f"rows={len(main)}, independent instances={len(main_instances)}")
    wa_pass = wa.full_single_flip_pass.astype("boolean").fillna(False)
    _check(rows, "wa_terminal_single_flip_check", bool(wa_pass.all()), f"{int(wa_pass.sum())}/{len(wa)} pass")
    _check(rows, "main_hard_constraint_violations", int(main.loc[main.algorithm.ne("all_offload"), "true_workload_violation"].sum()) == 0 and int(main.loc[main.algorithm.ne("all_offload"), "true_memory_violation"].sum()) == 0, "all feasible online, baseline, and QDP rows have zero true violations")

    binding = pd.read_csv(RUN_DIR / "binding_resource_runs.csv")
    pairs = set(zip(binding.workload_capacity_fraction, binding.memory_available_gb_config))
    required_pairs = set(product([0.2, 0.4, 0.6, 0.8, 1.0], [1.5, 2.0, 3.0, 5.0, 13.0]))
    hard = binding.loc[binding.memory_mode.eq("hard")]
    _check(rows, "binding_grid_coverage", len(binding) == 3750 and pairs == required_pairs and set(binding.memory_mode) == {"hard", "soft", "none"}, f"rows={len(binding)}, capacity pairs={len(pairs)}, modes={sorted(binding.memory_mode.unique())}")
    _check(rows, "binding_hard_constraints", int(hard.true_workload_violation.sum()) == 0 and int(hard.true_memory_violation.sum()) == 0 and bool(hard.full_single_flip_pass.fillna(False).all()), f"hard rows={len(hard)}, zero true violations, all single-flip checks pass")

    wireless = pd.read_csv(RUN_DIR / "wireless_sensitivity_runs.csv")
    required_wireless = {"bandwidth_mhz", "image_payload_mb", "uplink_power_w", "downlink_power_w", "path_loss_exponent"}
    _check(rows, "wireless_sensitivity_coverage", len(wireless) == 750 and set(wireless.factor) == required_wireless, f"rows={len(wireless)}, factors={sorted(wireless.factor.unique())}")
    robust = pd.read_csv(RUN_DIR / "profiling_error_robustness.csv")
    _check(rows, "profiling_error_coverage", len(robust) == 2400 and set(robust.error_magnitude) == {0.05, 0.1, 0.2, 0.3} and set(robust.protection) == {"no_margin", "fixed_margin", "conservative_quantization"}, f"rows={len(robust)}, four error magnitudes and three protections")
    scale = pd.read_csv(RUN_DIR / "scale_runs.csv")
    _check(rows, "scale_coverage", set(scale.n_nodes) == {30, 50, 80, 100, 150, 200} and len(scale) == 120, f"rows={len(scale)}, N={sorted(scale.n_nodes.unique())}")
    qdp = pd.read_csv(RUN_DIR / "scale_oracle_diagnostics.csv")
    _check(rows, "qdp_oracle_scope", bool(qdp.claim.eq("quantized_global_optimum_not_continuous_global_optimum").all()), "all QDP rows explicitly limited to the quantized problem")

    expected_tables = ["table_i_symbols.csv", "table_ii_profiler_metrics.csv", "table_iii_algorithm_comparison.csv"]
    _check(rows, "paper_tables_exist", all((TABLE_DIR / name).exists() and len(pd.read_csv(TABLE_DIR / name)) > 0 for name in expected_tables), "Table I-III are nonempty")
    figure_ok = True
    figure_detail = []
    for directory, suffix in FIGURE_FORMATS.items():
        actual = sorted(path.name for path in (FIGURE_ROOT / directory).glob(f"*{suffix}"))
        expected = sorted(f"{stem}{suffix}" for stem in EXPECTED_FIGURES)
        figure_ok = figure_ok and actual == expected
        figure_detail.append(f"{directory}:{len(actual)}")
    _check(rows, "six_figures_in_four_formats", figure_ok, ", ".join(figure_detail))
    pixels = pd.read_csv(AUDIT_DIR / "figure_pixel_audit.csv")
    _check(rows, "raster_figure_pixel_audit", len(pixels) == 12 and bool(pixels.nonblank_pass.all()) and bool(pixels.resolution_pass.all()), "12 PNG/TIFF audits are nonblank and high resolution")

    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
    ordered = ["### V-A.", "### V-B.", "### V-C.", "### V-D.", "### V-E.", "### V-F.", "### V-G.", "### V-H."]
    readme_ok = all(item in readme for item in ordered) and [readme.index(item) for item in ordered] == sorted(readme.index(item) for item in ordered) and "![" not in readme
    _check(rows, "readme_pdf_order_and_no_embedded_images", readme_ok, "I-VI and ordered V-A through V-H; figures referenced by links only")
    return {"overall_pass": all(row["status"] == "PASS" for row in rows), "checks": rows}


def main() -> None:
    audit = build_audit()
    write_csv(AUDIT_DIR / "acceptance_audit.csv", pd.DataFrame(audit["checks"]))
    write_json(AUDIT_DIR / "acceptance_audit.json", audit)
    write_json(
        AUDIT_DIR / "run_summary.json",
        {
            "experiment": "Mark10 standalone workload-aware wireless edge offloading",
            "acceptance_pass": audit["overall_pass"],
            "fresh_deepseek_profiles": 500,
            "task_pools": 5,
            "wireless_instances_per_pool": 10,
            "main_algorithm_rows": 1350,
            "binding_rows": 3750,
            "figures": EXPECTED_FIGURES,
            "formats": list(FIGURE_FORMATS),
            "claim_boundary": "WA-MCBR is a finite-descent single-flip local optimum; QDP is global only for the quantized problem.",
        },
    )
    if not audit["overall_pass"]:
        raise SystemExit("Acceptance audit failed; see 06_审计与复现/acceptance_audit.csv")
    print("Acceptance audit: PASS")


if __name__ == "__main__":
    main()
