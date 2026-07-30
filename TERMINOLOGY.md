# Terminology and Notation Ledger

本文档是后续 draft、图表和说明文件的唯一术语标准。优先级为：原始 PDF 明确定义的符号优先；PDF 未定义的符号使用本文档约定；Python 内部字段仅作为实现映射，不替代论文符号。

## 1. PDF-defined main notation

| Canonical symbol | Definition | Do not use in manuscript |
|---|---|---|
| `x_i` | binary offloading decision of task `i`; 1 is offload and 0 is local | `s_i` |
| `q_i` | mean-normalized workload of task `i` | unqualified raw workload |
| `v_i` | simulated GPU-memory demand of task `i` | measured VRAM demand |
| `K(x)` | number of offloaded tasks | bare `K` when the state dependence matters |
| `W(x)` | aggregate offloaded workload | bare `W` when the state dependence matters |
| `V(x)` | aggregate GPU-memory usage | bare `V` when the state dependence matters |
| `C_w` | nominal edge workload capacity | `C_W`, `W_max` |
| `V_avail` | available edge GPU memory | `V_max` |
| `B_total` | total wireless bandwidth | unqualified `B` in system equations |
| `J(x)` | barrier-free system objective used for common evaluation | `J(s)`, public business objective |

The hard constraints are written as:

```math
W(\mathbf{x})\leq(1-\varepsilon)C_w,
\qquad
V(\mathbf{x})\leq V_{\mathrm{avail}}.
```

## 2. Extended notation

| Canonical symbol | Definition | Status |
|---|---|---|
| `q_i^Count` | Count workload; equal to 1 for every task | baseline representation |
| `q_i^Data` | mean-normalized workload derived from observed execution time | prediction target |
| `q_i^LLM` | mean-normalized workload estimated by the LLM profiler | predicted representation |
| `m_i` | memory multiplier returned by the LLM profiler | simulated input |
| `v_base` | simulated base GPU-memory demand | assumed parameter |
| `B_i(x)` | bandwidth share `B_total/K(x)` | PDF wireless model |
| `R_i^up(K)` | uplink rate of task `i` | PDF wireless model |
| `R_i^down(K)` | downlink rate of task `i` | symmetric-link extension used by the simulation |
| `T_i^tx(K)` | wireless transmission delay of task `i` | model output |
| `E_i^tx(K)` | device transmission energy of task `i` | model output |
| `rho(x)` | workload utilization `W(x)/C_w` | congestion state |
| `D_q(W)` | monotone workload-congestion proxy | not a measured queue-delay predictor |
| `C_i^loc` | local execution cost | normalized cost |
| `C_i^off(K)` | base offloading cost under `K` offloaded tasks | normalized cost |
| `J_soft(x)` | soft-memory decision objective, equal to `J(x)` plus the barrier penalty | decision-only objective |
| `epsilon_alg` | strict-update tolerance of WA-MCBR | algorithm parameter |
| `Delta_q`, `Delta_v` | upward quantization steps for workload and memory | QDP parameters |

## 3. Canonical terminology

| Canonical term | First-use definition | Avoid |
|---|---|---|
| LLM profiler | replaceable offline semantic resource profiler; implemented with DeepSeek in this experiment | treating DeepSeek as the system architecture |
| Data workload | observed, execution-time-derived prediction target | Data model or Data predictor |
| Count workload | constant workload baseline | task count as ground truth |
| LLM workload | workload estimate produced by the LLM profiler | DeepSeek workload as a universal term |
| simulated GPU-memory demand | `v_i` constructed from `v_base` and `m_i` | measured per-request VRAM |
| workload-congestion proxy | `D_q(W)` | real queue-delay predictor |
| system objective | `J(x)`, excluding the soft barrier | public business objective |
| QDP-Oracle | offline optimum of the upward-quantized discrete problem | global optimum of the original continuous problem |
| single-flip local optimum | terminal WA-MCBR strategy after a full-node flip audit | globally optimal strategy |
| controller-assisted distributed updating | controller maintains aggregate state and assists marginal-cost evaluation | fully distributed implementation |

## 4. Implementation-field mapping

Existing Python identifiers and raw CSV headers remain stable for reproducibility. Interpret them as follows when writing the paper:

| Implementation field | Manuscript notation |
|---|---|
| `strategy`, `strategy_bits` | decision vector `x` |
| `k_offload` | `K(x)` |
| `estimated_workload`, `true_workload` | estimated/observed `W(x)` |
| `workload_capacity` | `C_w` |
| `workload_limit` | `(1-epsilon)C_w` |
| `estimated_memory_gb`, `true_memory_gb` | estimated/observed `V(x)` |
| `memory_available_gb` | `V_avail` |
| `public_objective_J` | common system objective `J(x)` |
| `decision_objective` | `J(x)` for hard/none modes or `J_soft(x)` for soft mode |
| `queue_delay_ms_per_offloaded_task` | sampled value of the proxy `D_q(W)` |
| internal model key `deepseek` | LLM profiler implemented with DeepSeek |

## 5. Claim boundaries

- WA-MCBR convergence means finite descent to a single-flip local optimum, not global optimality.
- QDP-Oracle is globally optimal only for the upward-quantized discrete problem.
- GPU-memory violations are software-simulation violations, not measured GPU OOM events.
- Wireless results come from a simulated single-cell symmetric-link approximation, not operator field measurements.
- LLM-profiler results are offline profiling results and do not establish real-time RAN coordination latency.
