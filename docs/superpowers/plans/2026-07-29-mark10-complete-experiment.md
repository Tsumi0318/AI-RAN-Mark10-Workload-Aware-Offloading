# Mark10 Complete Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a fresh, standalone Mark10 experiment that produces the complete data, algorithms, V-A through V-H evaluation, Fig. 1-6, Table I-III, README, and reproducibility audits required by the approved design.

**Architecture:** Mark10 is a Python package under `01_源码与配置/mark10` with separate modules for data selection, semantic profiling, workload validation, simulation, algorithms, experiment orchestration, plotting, and audits. Every result is written from fresh Mark10 inputs; Mark9 is read only as a behavioral reference, and no Mark9 result or semantic prediction is imported.

**Tech Stack:** Python 3.11+, numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, OpenAI-compatible DeepSeek client, pytest.

## Global Constraints

- Preserve `/Users/royzhangair/Desktop/AI RAN/Mark9` unchanged.
- Use exactly five disjoint 100-task pools and ten independent wireless instances per pool.
- Make exactly one successful fresh DeepSeek profile per selected task, with zero Mark9 cache reuse.
- Never persist or print the DeepSeek API key.
- Treat observed execution time as the Data workload reference; never include it in the DeepSeek intent.
- Treat per-task VRAM and all wireless quantities as explicitly labeled simulation values.
- Claim WA-MCBR only as a finite-descent single-flip local optimum.
- Claim QDP only as the global optimum of the quantized problem.
- Generate only Fig. 1-6 as high-resolution PNG; generate no supplementary figure and no report PDF.
- README Section V must include V-A through V-H in order, refer to figures without embedding them, and state numerical results and boundaries.

---

### Task 1: Repository scaffold, configuration, and source-data provenance

**Files:**
- Create: `00_原始数据/GenTD26/*`
- Create: `01_源码与配置/config.json`
- Create: `01_源码与配置/requirements.txt`
- Create: `01_源码与配置/mark10/__init__.py`
- Create: `01_源码与配置/mark10/io_utils.py`
- Create: `tests/test_io_and_config.py`

**Interfaces:**
- Consumes: approved design document and Mark9's official GenTD26 source files.
- Produces: `load_config() -> dict`, `sha256_file(path) -> str`, `write_csv(path, frame)`, `write_json(path, value)`, and a self-contained source-data directory.

- [ ] **Step 1: Write failing configuration and provenance tests**

```python
def test_config_defines_complete_protocol():
    config = load_config()
    assert config["n_task_pools"] == 5
    assert config["tasks_per_pool"] == 100
    assert config["wireless_seeds_per_pool"] == 10
    assert config["figures"] == [1, 2, 3, 4, 5, 6]

def test_write_csv_rejects_empty_frame(tmp_path):
    with pytest.raises(ValueError):
        write_csv(tmp_path / "empty.csv", pd.DataFrame())
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `PYTHONPATH=01_源码与配置 pytest tests/test_io_and_config.py -v`

Expected: collection fails because `mark10.io_utils` does not exist.

- [ ] **Step 3: Create configuration and minimal I/O implementation**

```python
def write_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
```

Include fixed seeds, task-pool counts, radio parameters, workload and memory grids, algorithm tolerances, signaling field sizes, API retry policy, and Fig. 1-6 filenames in `config.json`.

- [ ] **Step 4: Copy official GenTD26 files and write their manifest**

Run: `rsync -a --exclude '.DS_Store' ../Mark9/00_原始数据/GenTD26/ 00_原始数据/GenTD26/`

Run: `PYTHONPATH=01_源码与配置 python -m mark10.io_utils --manifest`

Expected: `06_审计与复现/source_data_manifest.csv` contains size and SHA-256 for every source file.

- [ ] **Step 5: Run tests and commit**

Run: `PYTHONPATH=01_源码与配置 pytest tests/test_io_and_config.py -v`

Expected: PASS.

Commit: `git commit -am "feat: scaffold Mark10 experiment and source provenance"`

---

### Task 2: Independent task pools and fresh DeepSeek profiling

**Files:**
- Create: `01_源码与配置/mark10/data_pipeline.py`
- Create: `01_源码与配置/mark10/semantic.py`
- Create: `tests/test_data_pipeline.py`
- Create: `tests/test_semantic.py`
- Generate: `02_任务池与画像/task_pool_01_pre_llm.csv` through `task_pool_05_pre_llm.csv`
- Generate: `02_任务池与画像/semantic_intents.csv`
- Generate: `02_任务池与画像/deepseek_resource_profiles.csv`
- Generate: `06_审计与复现/deepseek_generation_manifest.json`

**Interfaces:**
- Consumes: `load_config()`, GenTD26 request trace, and `DEEPSEEK_API_KEY` from the process environment.
- Produces: `build_task_pools(raw, config) -> list[pd.DataFrame]`, `build_intent(row) -> dict`, `parse_profile(text) -> SemanticProfile`, and `run_fresh_profiles(tasks, client, config) -> tuple[pd.DataFrame, dict]`.

- [ ] **Step 1: Write failing task-pool tests**

```python
def test_task_pools_are_disjoint_and_complete(raw_requests, config):
    pools = build_task_pools(raw_requests, config)
    assert [len(pool) for pool in pools] == [100] * 5
    ids = pd.concat(pools).source_row
    assert ids.nunique() == 500

def test_pools_cover_all_available_task_types(raw_requests, config):
    selected = pd.concat(build_task_pools(raw_requests, config))
    assert {"TXT_2_IMG", "IMG_2_IMG", "INPAINTING"}.issubset(set(selected.predict_type))
```

- [ ] **Step 2: Verify task-pool tests fail**

Run: `PYTHONPATH=01_源码与配置 pytest tests/test_data_pipeline.py -v`

Expected: FAIL because `build_task_pools` is missing.

- [ ] **Step 3: Implement deterministic date-stratified selection**

```python
usable = raw.loc[
    raw.predict_status.eq("SUCCEED")
    & raw.exec_time_seconds.gt(0)
    & raw.num_inference_steps.notna()
].copy()
usable["source_date"] = pd.to_datetime(usable.gmt_create).dt.date.astype(str)
```

Allocate disjoint source dates or non-overlapping windows to the five pools, stratify task types when possible, persist `source_row`, and assert global disjointness.

- [ ] **Step 4: Write failing semantic privacy and parser tests**

```python
def test_intent_does_not_leak_execution_time(task_row):
    intent = build_intent(task_row)
    serialized = json.dumps(intent)
    assert "exec_time" not in serialized
    assert "execution" not in serialized

def test_parser_rejects_out_of_range_multiplier():
    with pytest.raises(ValueError):
        parse_profile('{"compute_multiplier":9,"memory_multiplier":1,"semantic_class":"x","warning":"x"}')
```

- [ ] **Step 5: Verify semantic tests fail, then implement strict parsing and retries**

Run: `PYTHONPATH=01_源码与配置 pytest tests/test_semantic.py -v`

Expected: FAIL because semantic interfaces are missing.

Implement a client that sends only task type, prompt lengths, steps, image count, and LoRA count; validates finite multipliers in `[0.25, 4.0]`; records each attempt; and never exposes the API key.

- [ ] **Step 6: Run all 500 fresh DeepSeek calls**

Run with the key supplied only as an environment variable:

```bash
DEEPSEEK_API_KEY='process-only-secret' PYTHONPATH=01_源码与配置 \
python -m mark10.semantic --fresh
```

Expected manifest conditions: `selected_tasks=500`, `successful_profiles=500`, `fresh_api_calls=500`, `cache_hits=0`, `failed_profiles=0`, and no key field.

- [ ] **Step 7: Run tests, audit hashes, and commit**

Run: `PYTHONPATH=01_源码与配置 pytest tests/test_data_pipeline.py tests/test_semantic.py -v`

Expected: PASS.

Commit: `git add 01_源码与配置 02_任务池与画像 06_审计与复现 tests && git commit -m "feat: create independent pools and fresh DeepSeek profiles"`

---

### Task 3: Workload profiler validation and task-profile assembly

**Files:**
- Create: `01_源码与配置/mark10/profiler.py`
- Create: `tests/test_profiler.py`
- Generate: `02_任务池与画像/task_profiles.csv`
- Generate: `03_逐运行结果/profiler_predictions.csv`
- Generate: `04_汇总表格/table_ii_profiler_metrics.csv`
- Generate: `04_汇总表格/workload_distribution_summary.csv`

**Interfaces:**
- Consumes: five task pools and 500 DeepSeek profiles.
- Produces: `normalize_mean_one(values)`, `build_data_workload(exec_time)`, `cross_pool_predictions(tasks)`, and `profiler_metrics(truth, prediction)`.

- [ ] **Step 1: Write failing normalization and leakage-safe split tests**

```python
def test_each_pool_workload_has_mean_one(task_profiles):
    for _, pool in task_profiles.groupby("task_pool_id"):
        for column in ["q_count", "q_data", "q_llm"]:
            assert pool[column].mean() == pytest.approx(1.0)

def test_cross_pool_validation_never_trains_on_test_pool(tasks):
    predictions = cross_pool_predictions(tasks)
    assert not (predictions.train_pool_ids.str.contains(predictions.test_pool_id)).any()
```

- [ ] **Step 2: Verify RED and implement profile assembly**

Run: `PYTHONPATH=01_源码与配置 pytest tests/test_profiler.py -v`

Expected: FAIL because profiler functions are missing.

Implement Data workload as observed execution time divided by its pool mean, Count as ones, and LLM workload as compute multiplier divided by its pool mean. Set simulated task VRAM to `v_base_gb * memory_multiplier`.

- [ ] **Step 3: Implement cross-pool linear and tree baselines**

```python
features = [
    "prompt_length", "negative_prompt_length", "num_inference_steps",
    "num_images_per_prompt", "num_lora", "predict_type",
]
models = {
    "linear": Pipeline([... , LinearRegression()]),
    "tree": HistGradientBoostingRegressor(random_state=config["seed"]),
}
```

Use leave-one-pool-out evaluation and report MAE, RMSE, R-squared, and Spearman for Count, linear, tree, and DeepSeek predictions.

- [ ] **Step 4: Generate Table II source data and verify**

Run: `PYTHONPATH=01_源码与配置 python -m mark10.profiler`

Expected: 500 prediction rows and one held-out metric row per model per test pool plus overall summaries.

Run: `PYTHONPATH=01_源码与配置 pytest tests/test_profiler.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

Commit: `git add 01_源码与配置 02_任务池与画像 03_逐运行结果 04_汇总表格 tests && git commit -m "feat: validate workload profilers across independent pools"`

---

### Task 4: Shared system model, feasibility, and algorithms

**Files:**
- Create: `01_源码与配置/mark10/model.py`
- Create: `01_源码与配置/mark10/algorithms.py`
- Create: `tests/test_radio_and_model.py`
- Create: `tests/test_algorithms.py`
- Create: `tests/test_oracle.py`

**Interfaces:**
- Consumes: task profiles, wireless seeds, and configuration.
- Produces: `Scenario`, `generate_wireless_instance`, `run_wa_mcbr`, `swap_improve`, `capacity_greedy`, `lagrangian_relaxation`, `legacy_count_br`, `random_feasible`, `quantized_oracle`, and `exhaustive_quantized_oracle`.

- [ ] **Step 1: Write failing task-type payload and wireless reproducibility tests**

```python
def test_txt2img_has_no_input_image_uplink(txt2img_scenario):
    assert txt2img_scenario.input_image_uplink_bits[0] == 0

def test_img2img_uploads_input_and_downloads_output(img2img_scenario):
    assert img2img_scenario.uplink_bits[0] > img2img_scenario.metadata_prompt_bits[0]
    assert img2img_scenario.downlink_bits[0] > 0

def test_wireless_seed_changes_positions_but_not_tasks(tasks, config):
    a = generate_wireless_instance(tasks, 1, config)
    b = generate_wireless_instance(tasks, 2, config)
    assert not np.allclose(a.distance_m, b.distance_m)
    assert a.source_row.equals(b.source_row)
```

- [ ] **Step 2: Verify RED and implement the radio/system model**

Run: `PYTHONPATH=01_源码与配置 pytest tests/test_radio_and_model.py -v`

Expected: FAIL because model interfaces are missing.

Implement shared bandwidth, separate uplink/downlink powers, task-type payloads, workload congestion, hard workload and memory constraints, end-to-end latency, energy, and the barrier-free public objective.

- [ ] **Step 3: Write failing WA-MCBR feasibility and descent tests**

```python
def test_accepted_updates_strictly_reduce_public_objective(small_scenario):
    result = run_wa_mcbr(small_scenario, seed=7, record_trace=True)
    accepted = [r for r in result.trace if r["accepted"]]
    assert all(r["objective_after"] < r["objective_before"] - 1e-9 for r in accepted)
    assert full_single_flip_check(small_scenario, result.strategy, 1e-9)

def test_primary_repair_returns_jointly_feasible_strategy(small_scenario):
    repaired = benefit_resource_repair(small_scenario, np.ones(small_scenario.n, dtype=np.int8))
    assert small_scenario.feasible(repaired)
```

- [ ] **Step 4: Verify RED and implement online algorithms**

Run: `PYTHONPATH=01_源码与配置 pytest tests/test_algorithms.py -v`

Expected: FAIL because algorithm interfaces are missing.

Implement WA-MCBR, swap, capacity-aware greedy, Lagrangian relaxation with projected nonnegative multipliers and feasibility recovery, legacy, random, all-local, all-offload, signaling byte accounting, and both repair rules.

- [ ] **Step 5: Write failing conservative QDP and exhaustive-audit tests**

```python
def test_qdp_uses_upward_quantization(small_scenario):
    result = quantized_oracle(small_scenario, delta_q=0.1, delta_v=0.1)
    assert np.array_equal(result.q_integer, np.ceil(small_scenario.q / 0.1).astype(int))

def test_qdp_matches_exhaustive_quantized_optimum(tiny_scenario):
    qdp = quantized_oracle(tiny_scenario, 0.1, 0.1)
    brute = exhaustive_quantized_oracle(tiny_scenario, 0.1, 0.1)
    assert qdp.quantized_objective == pytest.approx(brute.quantized_objective)
```

- [ ] **Step 6: Verify RED, implement QDP, then run full model tests**

Run: `PYTHONPATH=01_源码与配置 pytest tests/test_oracle.py -v`

Expected: FAIL before implementation and PASS after sparse-state QDP, dominance pruning, K enumeration, and strategy recovery are implemented.

Run: `PYTHONPATH=01_源码与配置 pytest tests/test_radio_and_model.py tests/test_algorithms.py tests/test_oracle.py -v`

Expected: PASS.

- [ ] **Step 7: Commit**

Commit: `git add 01_源码与配置 tests && git commit -m "feat: implement workload-aware model algorithms and oracle"`

---

### Task 5: Complete V-A through V-H experiment orchestration

**Files:**
- Create: `01_源码与配置/mark10/experiments.py`
- Create: `01_源码与配置/run_mark10.py`
- Create: `tests/test_experiment_contracts.py`
- Generate: all CSV and JSON files under `03_逐运行结果/`

**Interfaces:**
- Consumes: task profiles, scenarios, algorithms, and configuration.
- Produces: `run_main_comparison`, `run_binding_grid`, `run_memory_model_comparison`, `run_wireless_sensitivity`, `run_profile_error_robustness`, `run_scale_analysis`, and `run_failure_case_audit`.

- [ ] **Step 1: Write failing matrix-coverage tests**

```python
def test_main_matrix_contains_50_independent_instances(main_runs):
    instances = main_runs[["task_pool_id", "wireless_seed"]].drop_duplicates()
    assert len(instances) == 50

def test_binding_grid_contains_every_capacity_pair(binding_runs):
    pairs = set(zip(binding_runs.workload_capacity_fraction, binding_runs.memory_available_gb))
    assert pairs == set(product([.2, .4, .6, .8, 1.0], [1.5, 2, 3, 5, 13]))
```

- [ ] **Step 2: Verify RED and implement resumable stage runners**

Run: `PYTHONPATH=01_源码与配置 pytest tests/test_experiment_contracts.py -v`

Expected: FAIL because experiment interfaces and fixtures are missing.

Each stage writes atomic checkpoint CSV files keyed by configuration hash, pool, wireless seed, scenario, algorithm, and algorithm seed. Existing Mark10 rows may resume only when their hash matches; Mark9 files are never read.

- [ ] **Step 3: Run V-C main comparison**

Run: `PYTHONPATH=01_源码与配置 python 01_源码与配置/run_mark10.py --stage main`

Expected: complete rows for three resource scenarios, 50 independent instances, and every listed algorithm; all hard-constraint algorithms have zero true violations.

- [ ] **Step 4: Run V-D binding grid and memory-model comparison**

Run: `PYTHONPATH=01_源码与配置 python 01_源码与配置/run_mark10.py --stage binding`

Expected: all 25 capacity pairs, common barrier-free objective fields, separate barrier penalty, utilization, violation, rejection reason, and Oracle gap.

- [ ] **Step 5: Run V-E wireless sensitivity and fairness**

Run: `PYTHONPATH=01_源码与配置 python 01_源码与配置/run_mark10.py --stage wireless`

Expected: bandwidth, payload, uplink power, downlink power, path-loss, and distance sweeps with near/middle/far metrics and Jain fairness.

- [ ] **Step 6: Run V-F profiling-error robustness**

Run: `PYTHONPATH=01_源码与配置 python 01_源码与配置/run_mark10.py --stage robustness`

Expected: plus/minus 5%, 10%, 20%, and 30% results for no margin, fixed margin, and conservative quantization, evaluated against unperturbed simulation truth.

- [ ] **Step 7: Run V-G scale, convergence, runtime, and signaling**

Run: `PYTHONPATH=01_源码与配置 python 01_源码与配置/run_mark10.py --stage scale`

Expected: N=30, 50, 80, 100, 150, and 200; online metrics separate from QDP state, pruning, runtime, and peak-memory diagnostics.

- [ ] **Step 8: Run V-H failure-case audit and experiment tests**

Run: `PYTHONPATH=01_源码与配置 python 01_源码与配置/run_mark10.py --stage failures`

Run: `PYTHONPATH=01_源码与配置 pytest tests/test_experiment_contracts.py -v`

Expected: PASS and a table containing extreme resources, large profile errors, far-user suppression, and large local-to-Oracle gaps.

- [ ] **Step 9: Commit**

Commit: `git add 01_源码与配置 03_逐运行结果 tests && git commit -m "feat: run complete Mark10 performance evaluation"`

---

### Task 6: Summary tables and statistical reporting

**Files:**
- Create: `01_源码与配置/mark10/summarize.py`
- Create: `tests/test_statistics_and_tables.py`
- Generate: `04_汇总表格/table_i_symbols.csv`
- Generate: `04_汇总表格/table_ii_profiler_metrics.csv`
- Generate: `04_汇总表格/table_iii_algorithm_comparison.csv`
- Generate: all V-A through V-H supporting summary CSV files.

**Interfaces:**
- Consumes: raw experiment-run CSV files.
- Produces: `summarize_with_ci(frame, keys, metrics)`, `jain_index(values)`, and paper table builders.

- [ ] **Step 1: Write failing confidence-interval and fairness tests**

```python
def test_jain_index_is_one_for_equal_values():
    assert jain_index(np.array([2.0, 2.0, 2.0])) == pytest.approx(1.0)

def test_summary_counts_independent_instances_not_update_seeds(main_runs):
    summary = summarize_with_ci(main_runs, ["scenario", "algorithm"], ["objective_J"])
    assert summary.independent_instances.min() == 50
```

- [ ] **Step 2: Verify RED, implement statistics, and build tables**

Run: `PYTHONPATH=01_源码与配置 pytest tests/test_statistics_and_tables.py -v`

Expected: FAIL before implementation and PASS after finite-value validation, independent-instance aggregation, 95% confidence intervals, and Jain calculations are implemented.

- [ ] **Step 3: Generate all tables and inspect schemas**

Run: `PYTHONPATH=01_源码与配置 python -m mark10.summarize`

Expected: Table I-III and all V-section summaries are nonempty and contain units, sample counts, mean, standard deviation, and 95% interval fields where applicable.

- [ ] **Step 4: Commit**

Commit: `git add 01_源码与配置 04_汇总表格 tests && git commit -m "feat: produce Mark10 paper tables and statistics"`

---

### Task 7: Fig. 1-6 publication plotting and visual QA

**Files:**
- Create: `01_源码与配置/mark10/plotting.py`
- Create: `tests/test_figure_contracts.py`
- Generate: `05_论文图表/PNG/Fig_1_system_model.png`
- Generate: `05_论文图表/PNG/Fig_2_workload_validation.png`
- Generate: `05_论文图表/PNG/Fig_3_binding_gap_heatmap.png`
- Generate: `05_论文图表/PNG/Fig_4_memory_models.png`
- Generate: `05_论文图表/PNG/Fig_5_wireless_fairness.png`
- Generate: `05_论文图表/PNG/Fig_6_convergence_runtime_signaling.png`
- Generate: `06_审计与复现/figure_pixel_audit.csv`

**Interfaces:**
- Consumes: exact source CSVs under `03_逐运行结果` and `04_汇总表格`.
- Produces: six and only six high-resolution PNG figures plus a pixel/dimension audit.

- [ ] **Step 1: Write failing figure contract tests**

```python
def test_exactly_six_required_figures_exist(output_dir):
    names = sorted(path.name for path in output_dir.glob("*.png"))
    assert names == EXPECTED_FIGURE_NAMES

def test_figures_are_nonblank_and_high_resolution(output_dir):
    for path in output_dir.glob("*.png"):
        image = Image.open(path).convert("RGB")
        assert image.width >= 1800 and image.height >= 1000
        assert np.asarray(image).std() > 5
```

- [ ] **Step 2: Verify RED, then implement plotting contracts**

Run: `PYTHONPATH=01_源码与配置 pytest tests/test_figure_contracts.py -v`

Expected: FAIL because figures and plotting code do not exist.

Before plotting, record for each figure its main conclusion, evidence panels, source CSV, axis units, and prohibited overclaims. Use the saved Python nature-figure backend and one restrained, colorblind-safe visual system.

- [ ] **Step 3: Generate Fig. 1-6**

Run: `PYTHONPATH=01_源码与配置 python -m mark10.plotting`

Expected: exactly six PNG files, no supplementary files, no PDF/SVG/TIFF outputs, and all panel labels and units present.

- [ ] **Step 4: Run automated and visual QA**

Run: `PYTHONPATH=01_源码与配置 pytest tests/test_figure_contracts.py -v`

Expected: PASS.

Open every PNG at original resolution and verify no clipping, overlap, unreadable type, empty panel, incorrect legend, or unsupported conclusion. Save the audit results to `figure_pixel_audit.csv`.

- [ ] **Step 5: Commit**

Commit: `git add 01_源码与配置 05_论文图表 06_审计与复现 tests && git commit -m "feat: generate six paper figures with visual audits"`

---

### Task 8: README V-A through V-H, acceptance audit, and final reproducibility run

**Files:**
- Create: `README.md`
- Create: `01_源码与配置/mark10/audit.py`
- Create: `01_源码与配置/run_all.sh`
- Create: `tests/test_readme_and_acceptance.py`
- Generate: `06_审计与复现/acceptance_audit.csv`
- Generate: `06_审计与复现/acceptance_audit.json`
- Generate: `06_审计与复现/run_summary.json`

**Interfaces:**
- Consumes: approved design, configuration, all raw outputs, all summary tables, and Fig. 1-6.
- Produces: a complete paper-aligned README and machine-readable acceptance verdicts.

- [ ] **Step 1: Write failing README and acceptance tests**

```python
def test_readme_contains_performance_sections_in_order():
    text = Path("README.md").read_text(encoding="utf-8")
    headings = ["V-A", "V-B", "V-C", "V-D", "V-E", "V-F", "V-G", "V-H"]
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)

def test_readme_does_not_embed_images():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "![" not in text

def test_acceptance_requires_fresh_500_call_manifest(audit):
    assert audit["fresh_profiles_500"]
    assert audit["cache_hits_zero"]
```

- [ ] **Step 2: Verify RED and write README from actual results**

Run: `PYTHONPATH=01_源码与配置 pytest tests/test_readme_and_acceptance.py -v`

Expected: FAIL because README and audit outputs do not exist.

Write Sections I-VI in PDF order. For each V-A through V-H subsection, include the research question, data, settings, algorithms, randomization unit, formulas, actual numeric results, file references, `见 Fig. X` or table reference, interpretation, and explicit non-claim. Do not embed figures.

- [ ] **Step 3: Implement line-by-line acceptance audit**

```python
checks = {
    "fresh_profiles_500": manifest["successful_profiles"] == 500,
    "cache_hits_zero": manifest["cache_hits"] == 0,
    "hard_violations_zero": hard_runs.true_violation.sum() == 0,
    "single_flip_all_pass": wa_runs.full_single_flip_pass.all(),
    "six_figures_only": actual_figures == expected_figures,
}
```

Include directory completeness, task-pool disjointness, 50-instance coverage, profiler metrics, capacity-grid coverage, wireless factors, robustness levels, scale values, QDP exhaustive match, Table I-III, README headings, and figure pixel audits.

- [ ] **Step 4: Run full tests and acceptance audit**

Run: `PYTHONPATH=01_源码与配置 pytest tests -v`

Expected: all tests PASS.

Run: `PYTHONPATH=01_源码与配置 python -m mark10.audit`

Expected: every required audit row is `PASS` and process exits zero.

- [ ] **Step 5: Verify reproducibility entrypoint and final tree**

Run: `bash 01_源码与配置/run_all.sh --verify-existing`

Expected: validates source hashes, semantic manifest, result schemas, tables, figures, README, and acceptance audit without making API calls.

Run: `git status --short && git diff --check`

Expected: only intentional uncommitted final artifacts before the final commit, and no whitespace errors.

- [ ] **Step 6: Final commit**

Commit: `git add . && git commit -m "complete Mark10 paper experiment and outputs"`

