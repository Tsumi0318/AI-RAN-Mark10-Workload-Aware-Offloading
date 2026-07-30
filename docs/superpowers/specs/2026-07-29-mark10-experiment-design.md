# Mark10 Complete Paper Experiment Design

## 1. Goal

Build a fresh, standalone Mark10 experiment that implements and evaluates the framework in `Distributed Workload-Aware Offloading for Memory-Constrained Wireless Edge AI.pdf`. Mark9 remains unchanged and is used only as a code-pattern reference. Mark10 must generate fresh task pools, fresh LLM-profiler outputs through the DeepSeek implementation, fresh simulation runs, all required data tables, and only the six figures specified by the PDF.

## 2. Research Boundary

- The experiment models one cell, one edge GPU pool, and a static decision window.
- Real GenTD26 fields include task type, prompt lengths, inference steps, LoRA count, image count, execution time, and production service traces.
- Wireless positions, channel states, radio powers, and shadowing are parameterized simulations rather than measured RAN data.
- Task memory demand is a simulation parameter defined by a base demand multiplied by a fresh DeepSeek semantic multiplier. It is not a per-request measured VRAM label.
- DeepSeek is an offline, replaceable workload and memory profiler. Python evaluates the deterministic system objective and makes every algorithmic decision.
- WA-MCBR is claimed only as a finite-descent single-flip local optimum after the full-node check.
- QDP-Oracle is claimed only as a global optimum of the stated quantized problem, not the continuous original problem.

## 3. Data Protocol

### 3.1 Independent task pools

- Filter successful GenTD26 requests with finite, positive execution time and complete model features.
- Build five mutually disjoint task pools with 100 requests each.
- Draw pools from distinct dates or non-overlapping time windows.
- Preserve the available TXT2IMG, IMG2IMG, and Inpainting mix through stratification where sample counts permit.
- Persist selected source row identifiers and file hashes so disjointness and provenance can be audited.

### 3.2 Fresh DeepSeek profiling

- Call `https://api.deepseek.com` once for each of the 500 unique tasks.
- Read the API key only from a process environment variable. Never save or print it.
- Use temperature zero and require a strict JSON response containing normalized compute multiplier, memory multiplier, semantic class, and a concise warning/explanation.
- Do not send observed execution time to DeepSeek.
- Do not reuse Mark9 predictions or cache entries.
- Retry invalid or failed calls up to three times. If a task still has no valid response, record the failure and stop before the main experiment instead of substituting an old or random value.
- Save requested model, resolved model, latency, token counts, status, response hash, and output provenance.

### 3.3 Workload representations

- Count workload assigns one unit to every task.
- Data workload is the observed execution time normalized to mean one and serves as the trace-derived workload reference.
- LLM workload uses the fresh DeepSeek compute multiplier.
- Normalize Count, Data, and LLM workloads independently to mean one on each task pool before comparison.
- Use held-out normalized observed execution time as the workload-validation target. Predictive baselines and DeepSeek do not receive execution time as an input.

### 3.4 Memory and payloads

- Set task memory demand to `v_base * memory_multiplier`, and record `v_base` as an explicit simulation parameter.
- TXT2IMG uploads prompt and metadata and downloads generated images.
- IMG2IMG uploads prompt, metadata, and input image payload and downloads generated images.
- Inpainting uploads prompt, metadata, input image, and mask payload and downloads generated images.
- Image byte sizes are explicit simulation parameters because the trace does not contain compressed image resolution or payload bytes.

### 3.5 Wireless instances

- Generate ten independent wireless instances per task pool.
- Resample user position and shadowing for each instance.
- Keep each generated instance identical across algorithms.
- Model uplink and downlink powers separately.
- Persist distance, position, shadowing, path loss, channel gain, rates, and payload sizes.

## 4. Model and Algorithms

### 4.1 Shared model

All algorithms use the same binary decision vector `x`, aggregate count `K(x)`, aggregate workload `W(x)`, aggregate GPU-memory usage `V(x)`, radio model, workload-congestion proxy `D_q(W)`, workload limit `(1-epsilon)C_w`, memory limit `V_avail`, and system objective `J(x)`.

### 4.2 Algorithms

- WA-MCBR with full-objective marginal updates.
- WA-MCBR-Swap with feasible one-off/one-on exchanges.
- QDP-Oracle with conservative `ceil(q_i / Delta_q)` and `ceil(v_i / Delta_v)` quantization.
- Capacity-aware greedy.
- Lagrangian-relaxation baseline with documented multiplier updates, projection, feasibility recovery, and stopping conditions.
- Legacy count-based best response.
- Random feasible strategy.
- All-local strategy.
- All-offload strategy, explicitly marked infeasible where appropriate.

### 4.3 Initialization and repair

- The primary repair removes the selected task with the lowest estimated benefit per normalized joint-resource burden until feasible.
- The old largest-memory removal rule remains only as an initialization ablation.
- No algorithm may accept a candidate that violates the workload or memory hard constraint.

### 4.4 Signaling model

- Describe the implementation as controller-assisted distributed updating.
- Account for bytes used to broadcast aggregate `K`, `W`, and `V`, identify the selected task, return the decision, and acknowledge the update.
- Report payload bytes separately from any assumed protocol header bytes.
- Treat signaling as simulated control traffic rather than measured cellular signaling.

## 5. Experiment Matrix

### 5.1 Main comparison

- Run every online algorithm over five task pools and ten wireless instances.
- Separate algorithm update-order randomness from task-pool and wireless-instance randomness.
- Evaluate resource-abundant, moderately constrained, and highly constrained scenarios.
- Report objective, quantized Oracle gap, end-to-end delay, energy, offload rate, workload utilization, memory utilization, violations, runtime, updates, and signaling.

### 5.2 Workload profiler validation

- Split by task pool or time window to prevent leakage.
- Compare Count, linear regression, tree regression, and LLM-profiler predictions from the DeepSeek implementation.
- Report MAE, RMSE, R-squared, and Spearman rank correlation on the held-out test data.
- If DeepSeek does not outperform the supervised baselines, describe it as a heuristic replaceable profiler.

### 5.3 Binding resource constraints

- Sweep workload limits over 20%, 40%, 60%, 80%, and 100% of the base capacity.
- Sweep available memory over 1.5, 2, 3, 5, and 13 GB.
- Compare hard constraint, soft exponential barrier, and unconstrained memory models using the same system objective `J(x)` without the barrier penalty.
- Report barrier penalties separately.

### 5.4 Wireless sensitivity and fairness

- Sweep total bandwidth, image payload, uplink power, downlink power, path-loss exponent, and distance.
- Report near, middle, and far user results.
- Define Jain fairness over a documented per-user benefit or service-quality quantity before running the analysis.

### 5.5 Profiling-error robustness

- Perturb estimated workload and memory by plus/minus 5%, 10%, 20%, and 30%.
- Make decisions with perturbed estimates and evaluate violations against the unperturbed simulation truth.
- Compare no margin, fixed margin, and conservative quantization.
- Report objective degradation, strategy change rate, workload violation rate, and memory violation rate.

### 5.6 Scale, convergence, runtime, and signaling

- Evaluate `N` in 30, 50, 80, 100, 150, and 200 using the deterministic ordered union of task pools 1 and 2, so no additional semantic calls or reused Mark9 profiles are introduced.
- Record objective traces, updates, runtime, and control bytes.
- Report QDP grid, states created, states pruned, pruning rate, peak live states, runtime, and Python peak memory separately from online algorithms.

## 6. Required Outputs

Mark10 must contain every directory below:

```text
Mark10/
  00_原始数据/
  01_源码与配置/
  02_任务池与画像/
  03_逐运行结果/
  04_汇总表格/
  05_论文图表/PNG/
  06_审计与复现/
  tests/
  docs/
  README.md
```

The experiment must produce only the six figures specified by the PDF:

- Fig. 1: system model and K/W/V coupling.
- Fig. 2: workload-profile prediction scatter and workload CDF.
- Fig. 3: workload-capacity by memory-capacity Oracle-gap heatmap.
- Fig. 4: hard constraint, soft barrier, and unconstrained comparison.
- Fig. 5: wireless sensitivity, distance-stratified offloading, and fairness.
- Fig. 6: convergence, updates, runtime, and signaling.

No supplementary figures and no experiment-report PDF will be generated. Figures are high-resolution PNG files, and their source data are saved as CSV.

Required paper tables are:

- Table I: symbol definitions and units.
- Table II: workload-profiler validation metrics.
- Table III: main algorithm comparison.

## 7. README Contract

The README follows the PDF's Sections I-VI. Section V must reproduce V-A through V-H in the exact order. Every subsection must state:

- research question;
- data and sample construction;
- variable and fixed parameters;
- compared algorithms;
- number of runs and randomization unit;
- metrics and formulas;
- actual numerical results;
- corresponding CSV or JSON files;
- corresponding figure or table reference;
- evidence-constrained interpretation;
- what the result does not establish.

The README refers to figures using text such as `见 Fig. 3` and links the output filename, but does not embed or display images. V-C uses Table III. V-F and V-H use result tables and detailed text without extra figures.

## 8. Tests and Audits

- Test task-pool disjointness, stratification, and deterministic regeneration.
- Test no execution-time leakage into the DeepSeek intent.
- Test strict LLM response parsing and retry/failure behavior without storing the API key.
- Test workload mean-one normalization.
- Test task-type-specific uplink and downlink payloads.
- Test hard workload and memory feasibility.
- Test that every accepted WA-MCBR update strictly reduces `J` and that termination passes a full single-flip audit.
- Test conservative QDP quantization and compare small cases against exhaustive enumeration.
- Test that hard, soft, and unconstrained policies are evaluated with the same barrier-free system objective `J(x)`.
- Test wireless-instance independence and cross-algorithm identity.
- Test metric formulas, confidence intervals, Jain fairness, and signaling byte accounting.
- Audit every required CSV column, every required README subsection, and every required Fig. 1-6 output.
- Render and visually inspect all six PNG figures for legibility, clipping, label correctness, and nonempty panels.

## 9. Completion Criteria

Mark10 is complete only when:

- all 500 tasks have valid fresh DeepSeek responses and zero cache reuse;
- all required experiment stages have finished without missing instance combinations;
- all hard-constraint runs have zero true workload and memory violations;
- all reported WA-MCBR terminal strategies pass the full single-flip audit;
- small QDP instances match exhaustive enumeration;
- Fig. 1-6, Table I-III, raw run data, configurations, manifests, and audits exist;
- README Section V contains complete V-A through V-H text and correct result references;
- the full test and acceptance-audit commands exit successfully;
- every conclusion distinguishes simulation evidence, local optimality, quantized optimality, and real deployment evidence.
