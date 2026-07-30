# Workload-Aware Offloading for Memory-Constrained Wireless Edge AI

这是一次独立、可审计的 Mark10 仿真实验。它按 `Distributed Workload-Aware Offloading for Memory-Constrained Wireless Edge AI` 的 I-VI 结构实现共享无线链路和边缘 GPU 资源下的二元卸载，并将每项数值、图表和结论边界保存为可复查文件。

**先读结论边界。** GenTD26 提供真实生产生成式 AI 服务请求特征与执行时间；无线信道、端侧设备、图像载荷字节数、任务级显存和资源容量均为仿真设定。DeepSeek 在本实验中是离线、可替换的工作量与模拟显存画像器；Python 计算所有确定性代价并执行决策。WA-MCBR 的保证是有限下降后的单翻转局部最优，不是全局最优。QDP-Oracle 的“全局最优”只针对向上量化后的离散问题。

## Repository map

- [`00_原始数据`](00_原始数据)：GenTD26 原始追踪及 SHA-256 溯源。
- [`01_源码与配置`](01_源码与配置)：配置、Python 实现和验证入口。
- [`02_任务池与画像`](02_任务池与画像)：5 个独立任务池、语义 Intent、DeepSeek 输出与任务画像。
- [`03_逐运行结果`](03_逐运行结果)：每次仿真的原始记录。
- [`04_汇总表格`](04_汇总表格)：Table I-III 和 V-A 至 V-H 统计汇总。
- [`05_论文图表`](05_论文图表)：Fig. 1-6 的 [PNG](05_论文图表/PNG)、[PDF](05_论文图表/PDF)、[SVG](05_论文图表/SVG)、[TIFF](05_论文图表/TIFF) 版本，以及按 Table III 样式渲染的 [Table I-II PNG](05_论文图表/TABLE_PNG)。
- [`06_审计与复现`](06_审计与复现)：源数据、模型调用、图像与验收审计。

## I. Introduction

生成式 AI 请求的推理步数、Prompt 长度、LoRA 使用情况和执行时间不同；因此只用卸载任务数量 $`K`$ 描述中心压力会忽略工作量异构性。本实验研究单小区、单边缘 GPU 池、静态决策窗口中的二元选择：每个任务要么本地执行，要么通过共享无线链路卸载到边缘池。

问题不是“找一个已证明的全局最低成本调度”，而是：在工作量和显存硬约束下，能否得到可行、可复查、单边稳定的分布式决策，并用一个明确的离散 Oracle 对照其效率损失。

## II. Related Scope and Design Position

本实现对应工作量感知、显存约束、无线卸载和分布式单节点更新四个部分。它不等同于完整 AI-RAN 部署：没有真实端侧硬件、运营商无线测量、动态到达过程、多小区干扰或真实 GPU OOM 事件。

与只按任务数建模相比，中心拥塞由聚合工作量 $`W`$ 驱动；与软显存惩罚不同，主算法使用显式显存硬约束。为比较不同策略，公共业务目标不含软障碍项；软障碍只用于软约束决策模型，罚项单独记录。

## III. System Model and Problem Formulation

主要符号见 [Table I CSV](04_汇总表格/table_i_symbols.csv) 和 [Table I PNG](05_论文图表/TABLE_PNG/Table_I_main_notation.png)，扩展符号的单位和来源见 [详细符号表](04_汇总表格/iii_symbol_details.csv)，系统关系见 [Fig. 1](05_论文图表/PNG/Fig_1_system_model.png)。

### A. Task decisions and aggregate state

对任务集合 $`\mathcal{N}=\lbrace 1,\ldots,N\rbrace`$，令：

```math
s_i \in \{0,1\},\qquad
s_i=1\;\text{表示卸载，}\quad s_i=0\;\text{表示本地执行}.
```

共享状态为：

```math
K(\mathbf{s})=\sum_i s_i,
\qquad
W(\mathbf{s})=\sum_i s_iq_i,
\qquad
V(\mathbf{s})=\sum_i s_iv_i.
```

其中 $`q_i`$ 是均值归一化的任务工作量，$`v_i`$ 是任务级模拟显存需求。任务的真实执行时间只用于构造验证标签与确定性代价；它不发送给 DeepSeek。每条任务的显存按下式构造：

```math
v_i=v_{\mathrm{base}}m_i,
\qquad v_{\mathrm{base}}=0.1625\;\mathrm{GB},
```

其中 $`m_i`$ 是 DeepSeek 返回的显存倍率。因此 $`v_i`$ 是模拟值，不是逐请求实测 VRAM。

### B. Wireless transmission model

每个无线实例重采样位置和对数正态阴影衰落；同一实例在所有算法间保持一致。若当前有 $`K\geq1`$ 个卸载任务，则每个任务平均获得 $`B/K`$ 带宽：

```math
R_i^{\mathrm{up}}=\frac{B}{K}\log_2\left(1+\frac{P_{\mathrm{up}}g_i}{N_0B/K}\right),
\qquad
R_i^{\mathrm{down}}=\frac{B}{K}\log_2\left(1+\frac{P_{\mathrm{down}}g_i}{N_0B/K}\right).
```

TXT2IMG 仅上传 Prompt/元数据；IMG2IMG 额外上传输入图像；Inpainting 额外上传输入图像与 mask。下行均下载生成图像。传输时延和设备传输能耗为：

```math
T_i^{\mathrm{tx}}=\frac{L_i^{\mathrm{up}}}{R_i^{\mathrm{up}}}+\frac{L_i^{\mathrm{down}}}{R_i^{\mathrm{down}}},
\qquad
E_i^{\mathrm{tx}}=P_{\mathrm{up}}\frac{L_i^{\mathrm{up}}}{R_i^{\mathrm{up}}}+P_{\mathrm{rx}}\frac{L_i^{\mathrm{down}}}{R_i^{\mathrm{down}}}.
```

这些无线量和图像字节数是仿真参数，并非现场测量。

### C. Workload congestion and hard constraints

设边缘工作量容量为 $`C_W`$，使用率为 $`\rho=W/C_W`$。实现采用如下排队代理：

```math
D_{\mathrm{comp}}(W)=D_0+a\frac{\rho}{1-\rho},
\qquad \rho<1.
```

工作量安全上限取 $`W_{\max}=0.95C_W`$，显存可用上限为 $`V_{\max}`$。主算法只接受同时满足的策略：

```math
W(\mathbf{s})\leq W_{\max},
\qquad
V(\mathbf{s})\leq V_{\max}.
```

### D. Cost and public objective

令 $`t_i`$ 为数据中的执行时间，$`\tilde t`$ 为同一实例的中位执行时间。实现中的本地和卸载基础成本为：

```math
C_i^{\mathrm{loc}}=1.8\frac{t_i}{\tilde t}+0.4,
```

```math
C_i^{\mathrm{off,base}}=
\frac{t_i}{\tilde t}
 +0.5\frac{T_i^{\mathrm{tx}}}{1\;\mathrm{s}}
 +0.5\frac{E_i^{\mathrm{tx}}}{0.1\;\mathrm{J}}.
```

用于统一比较的无障碍公共业务目标为：

```math
J(\mathbf{s})=
\sum_{i:s_i=0}C_i^{\mathrm{loc}}
+\sum_{i:s_i=1}C_i^{\mathrm{off,base}}
+K(\mathbf{s})D_{\mathrm{comp}}(W(\mathbf{s})).
```

软显存策略仅在**决策阶段**附加：

```math
J_{\mathrm{soft}}(\mathbf{s})=J(\mathbf{s})+
K(\mathbf{s})\alpha\exp\left[\beta\left(\frac{V(\mathbf{s})}{V_{\max}}-1\right)\right],
```

其中 $`\alpha=1.2`$、$`\beta=8.0`$。V-D 报告 $`J`$ 与 barrier penalty 为不同字段，避免把不同目标混作性能比较。

### E. Quantized Dynamic-Programming Oracle

QDP 对 $`q_i`$ 和 $`v_i`$ 都向上量化，保证量化解对原资源约束保守：

```math
\bar q_i=\left\lceil\frac{q_i}{\Delta_q}\right\rceil,
\qquad
\bar v_i=\left\lceil\frac{v_i}{\Delta_v}\right\rceil,
\qquad
\Delta_q=0.1,\quad \Delta_v=0.1\;\mathrm{GB}.
```

它枚举卸载数量并在稀疏状态 $`(k,w,m)`$ 上进行动态规划、支配剪枝和策略恢复。QDP 是量化问题的全局最优，不是原连续问题的全局最优，也不作为在线算法 runtime 的对照。

## IV. Proposed Workload-Aware Offloading Algorithms

### A. Feasible initialization and WA-MCBR

WA-MCBR 从随机二元策略开始。若初始策略超出任何硬资源限制，则重复移除“公共目标收益 / 归一化联合资源负担”最低的已卸载任务，直到可行。每次异步抽取一个任务 $`i`$，分别评估 $`s_i=0`$ 与 $`s_i=1`$ 的完整决策目标。候选可行且严格降低目标时，更新规则为：

```math
s_i \leftarrow 1
\quad \mathrm{if}\quad
J_{\mathrm{decision}}(s_i=1)
<
J_{\mathrm{decision}}(s_i=0)-\epsilon.
```

```math
s_i \leftarrow 0
\quad \mathrm{if}\quad
J_{\mathrm{decision}}(s_i=0)
<
J_{\mathrm{decision}}(s_i=1)-\epsilon.
```

若两项均不严格更优，则保持 $`s_i`$ 不变。

其中 $`\epsilon=10^{-9}`$。连续 $`N`$ 次无变化后，代码还会显式检查所有单节点翻转；只有不存在有利且可行的单翻转时停止。因此该终态是本公共目标下的单翻转局部最优。

### B. Comparison algorithms

主比较使用 QDP-Oracle、WA-MCBR、WA-MCBR-Swap、容量感知 Greedy、Lagrangian relaxation、Legacy count-based BR、Random feasible、All-local 与 All-offload。All-offload 故意保留为违反约束的参考行；它不计入可行算法结论。

WA-MCBR-Swap 在单翻转停止后尝试一个任务卸载、一个任务本地的可行交换；它不改变 WA-MCBR 的局部最优含义。Lagrangian 方法使用投影非负乘子并在结束时进行可行修复。

### C. Controller-assisted signaling

每次 WA-MCBR 更新计入聚合 (K,W,V) 广播、任务标识、决策回复和确认四类控制消息。有效载荷为 38 bytes，按四条消息、每条 20-byte 协议头计算，总计 118 bytes/update。该项是模拟控制流量，不是实际蜂窝控制面测量。

## V. Performance Evaluation

所有统计表均将“任务池 × 无线实例”作为独立样本，而不是把算法更新顺序当作新的网络实例。每个均值行均在 5 个非重叠任务池和每池 10 个独立无线实例上计算，即 (n=50)，并在汇总 CSV 中给出标准差与 95% 置信区间。

### V-A. Experimental Setup

**研究问题。** 如何在不混淆真实追踪与仿真假设的条件下，构造可重复的工作量感知无线边缘卸载评估？

**数据与样本。** 从 GenTD26 成功请求中按不同日期构造 5 个互不重叠、每池 100 条的任务池：2024-11-20、2024-11-25、2024-11-28、2024-12-03、2024-12-08。每池至少含 75 条 TXT2IMG、20 条 IMG2IMG 和 2 条 Inpainting；500 条 `source_row` 全部唯一。每条任务均在本次实验中重新调用 DeepSeek 一次，得到 500/500 有效画像，0 cache hit，0 failed profile，temperature=0。实际解析模型为 `deepseek-v4-flash`，平均 API 时延 1043.82 ms、P95 1300.48 ms；它是离线画像开销，不计入在线 WA-MCBR runtime。

**固定与变化设置。** 每个任务池生成 10 个独立位置/阴影衰落实例。固定默认值为 $`B=20`$ MHz、$`P_{\mathrm{up}}=0.2`$ W、$`P_{\mathrm{down}}=1.0`$ W、路径损耗指数 3.5、基准工作量容量 60、$`v_{\mathrm{base}}=0.1625`$ GB。主比较使用 abundant $`(C_W\times1.0,13\mathrm{GB})`$、moderate $`(C_W\times0.6,5\mathrm{GB})`$ 和 highly constrained $`(C_W\times0.3,2\mathrm{GB})`$；V-D 额外覆盖 5 个工作量容量和 5 个显存容量的完整网格，因此包含单资源与双资源紧约束。

**指标。** 报告 (J)、投影后量化 Oracle gap、端到端时延、设备能耗、卸载率、工作量/显存利用率、真实违规率、runtime、updates 和 signaling。完整参数见 [V-A setup](04_汇总表格/v_a_experimental_setup.csv)，输入和画像见 [`02_任务池与画像`](02_任务池与画像)，调用证据见 [DeepSeek manifest](06_审计与复现/deepseek_generation_manifest.json)。

**证据边界。** 真实的是云端任务属性与执行时间；无线、端侧、载荷字节、显存和容量不是现场实测。该设置不能证明真实 AI-RAN 部署性能。

### V-B. Workload Representation Validation

**研究问题。** 任务级工作量比常数任务计数是否更有信息量？DeepSeek 画像是否达到可验证价值？

**协议。** 以执行时间除以各池均值得到 $`q_i^{\mathrm{data}}`$，Count 固定为 1，DeepSeek compute multiplier 也在池内归一化为均值 1。采用留一任务池验证：每次在其他 4 池训练线性与随机森林回归，在保留的第 5 池评估。指标为：

```math
\mathrm{MAE}=\frac{1}{n}\sum_i|\hat q_i-q_i|,
\qquad
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_i(\hat q_i-q_i)^2}.
```

同时报告 $`R^2`$ 和 Spearman 秩相关。

**实际结果。** 500 条池外预测的 Count / DeepSeek / Linear / Tree 分别为：MAE $`=0.46895/0.47615/0.37048/0.32511`$，RMSE $`=0.68826/0.67432/0.54884/0.48721`$，$`R^2=0.00000/0.04009/0.36410/0.49890`$，Spearman $`=0.00000/0.23470/0.47358/0.54763`$。Tree 是这里最强的受监督基线，DeepSeek 略优于 Count 的 RMSE 和 $`R^2`$，但 MAE 更高，且明显落后于受监督模型。

见 [Table II CSV](04_汇总表格/table_ii_profiler_metrics.csv)、[Table II PNG](05_论文图表/TABLE_PNG/Table_II_profiler_metrics.png)、[逐池指标](04_汇总表格/v_b_profiler_metrics_by_pool.csv) 与 [Fig. 2](05_论文图表/PNG/Fig_2_workload_validation.png)。

**受证据约束的结论。** 本数据上，任务级工作量确实含有异构信息；DeepSeek 只能定位为可替换的启发式 profiler，不能声称优于有标签监督回归，也不能用该结果推广到其他模型或工作负载。

### V-C. Overall Performance Comparison

**研究问题。** 在相同任务、无线实例和硬约束协议下，WA-MCBR 能否获得可行解并接近量化 QDP 对照？

**协议。** 三个资源场景各运行 50 个独立实例，9 种算法共 1,350 行原始记录。QDP gap 按策略先投影到保守量化可行空间后再计算；这样可避免原连续策略因向上量化而不可行的比较偏差。WA-MCBR 原策略有 97/150 个案例可直接量化可行，投影后 150/150 均可比较；平均投影改变率 0.433%，最大 2%。

**实际结果。** WA-MCBR 的平均 $`J`$ 在 abundant / moderate / highly constrained 场景分别为 245.2331 / 246.5251 / 254.4292；对应投影量化 gap 为 0.0093% / 0.0171% / 1.2189%，平均卸载率为 16.84% / 14.60% / 8.24%，显存利用率为 31.34% / 70.43% / 96.14%。所有 150 个 WA-MCBR 终态通过完整单翻转检查，真实工作量和显存违规均为 0。

在 highly constrained 场景，WA-MCBR-Swap 的公共目标为 251.1484，低于 WA-MCBR 的 254.4292；这说明单翻转局部终态仍可能有可改进的交换邻域。QDP 的连续公共目标并不必然最小，因为 QDP 优化的是量化目标；因此不应按连续 $`J`$ 将 QDP 误称为全局最优。

见 [Table III](04_汇总表格/table_iii_algorithm_comparison.csv)、[主比较原始记录](03_逐运行结果/main_algorithm_runs.csv) 和 [QDP diagnostics](03_逐运行结果/main_oracle_diagnostics.csv)。

**受证据约束的结论。** WA-MCBR 在这 150 个静态实例上都达到可行的单翻转局部最优，并且投影量化 gap 较小；这不是连续原问题的全局最优保证，也不能说明其总是优于所有启发式算法。

### V-D. Performance under Binding Workload and Memory Constraints

**研究问题。** 当工作量与显存约束实际绑定时，硬显存约束是否保持可行？硬、软、无显存约束在同一个公共业务目标下有何取舍？

**协议。** 对 5 个工作量容量比例 $`\{20\%,40\%,60\%,80\%,100\%\}`$ 与 5 个可用显存 $`\{1.5,2,3,5,13\}`$ GB 的 25 个组合，在 50 个独立实例上分别运行 hard、soft、none 三种显存策略，共 3,750 个记录。三种策略用同一个无障碍 $`J`$ 评估；soft barrier penalty 独立保存。

**实际结果。** hard 策略的 1,250 个记录均无真实工作量或显存违规，且全部通过单翻转检查；跨网格平均 $`J=251.1354`$、工作量利用率 30.85%、显存利用率 74.34%。soft 的平均 $`J=250.9165`$、平均显存利用率 60.31%、平均 barrier penalty 0.7603，真实显存违规为 0。none 的平均 $`J=247.5648`$，但真实显存违规率为 50.4%，平均显存利用率 115.71%。在最紧的 $`20\%`$ / 1.5 GB 条件下，hard 的平均投影量化 gap 为 0.9097%；其显存利用率为 95.98%。

见 [Fig. 3](05_论文图表/PNG/Fig_3_binding_gap_heatmap.png)、[Fig. 4](05_论文图表/PNG/Fig_4_memory_models.png)、[容量网格汇总](04_汇总表格/v_d_binding_capacity_summary.csv) 与 [逐运行记录](03_逐运行结果/binding_resource_runs.csv)。

**受证据约束的结论。** 本仿真内硬约束以较高的公共目标换取零违规；无约束策略的较低 $`J`$ 不能解释为更可部署，因为它经常超过模拟显存上限。显存“违规”是软件代理，不是实测 GPU OOM。

### V-E. Impact of Wireless Conditions

**研究问题。** 带宽、载荷、功率、传播损耗和距离如何改变卸载收益与用户公平性？

**协议。** 在 moderate 资源设置上，分别扫描带宽 10/20/40 MHz、图像载荷 0.1/0.5/1.0 MB、上行功率 0.1/0.2/0.4 W、下行功率 0.5/1.0/2.0 W、路径损耗指数 3.0/3.5/3.8。每点 50 个独立实例，共 750 个运行。距离以每实例三分位划为 near/middle/far。公平性采用：

```math
\mathcal{F}=\frac{(\sum_i z_i)^2}{N\sum_i z_i^2},
\qquad z_i=\frac{1}{1+D_i^{\mathrm{e2e}}}.
```

**实际结果。** 带宽从 10 增至 40 MHz 时，平均 $`J`$ 从 251.6856 降至 241.3656，卸载率从 11.00% 升至 18.00%，Jain 指数从 0.82631 升至 0.83698。图像载荷从 0.1 增至 1.0 MB 时，$`J`$ 从 240.6993 升至 252.9188，卸载率从 19.88% 降至 10.16%。在 20 MHz 基准下，near/middle/far 的卸载率分别为 22.80% / 12.32% / 8.58%，平均端到端时延为 41.10 / 44.35 / 45.41 s。路径损耗指数从 3.0 到 3.8 时，公平性从 0.83904 降到 0.82392。

见 [Fig. 5](05_论文图表/PNG/Fig_5_wireless_fairness.png)、[无线敏感性汇总](04_汇总表格/v_e_wireless_sensitivity_summary.csv)、[距离分层汇总](04_汇总表格/v_e_distance_layer_summary.csv) 与对应逐运行记录。

**受证据约束的结论。** 本无线模型表明带宽、载荷和距离会改变卸载决策与公平性；由于位置、阴影衰落、功率和路径损耗均为仿真输入，不能把数值直接解释为现场网络 KPI。

### V-F. Robustness to Profiling Errors

**研究问题。** 工作量与显存估计误差是否会导致公共目标退化或硬约束违规？

**协议。** 决策使用扰动估计：

```math
\hat q_i=q_i(1+\varepsilon_i^q),
\qquad
\hat v_i=v_i(1+\varepsilon_i^v),
```

其中 $`\varepsilon_i^q,\varepsilon_i^v\in\{\pm5\%,\pm10\%,\pm20\%,\pm30\%\}`$。比较 no margin、10% fixed margin 和 conservative quantization 三种保护，共 $`50\times4\times4\times3=2400`$ 个案例；评价时重新使用未扰动的真实模拟 $`q_i,v_i`$。

**实际结果。** 所有 2,400 个记录的真实工作量与显存违规率均为 0。汇总层面最大的平均目标退化出现在 30%、$`q`$ 低估、$`v`$ 高估、conservative quantization 条件，为 2.3136%，平均策略变化率 13.30%。单个最不利记录的目标退化为 6.6373%。

见 [鲁棒性汇总](04_汇总表格/v_f_profile_error_summary.csv)、[逐运行结果](03_逐运行结果/profiling_error_robustness.csv) 和 [失败案例表](04_汇总表格/v_h_failure_cases.csv)。

**受证据约束的结论。** 在当前安全余量和任务规模下，所测试的均匀乘性误差没有造成违规；这不意味着所有模型误差都安全，尤其不覆盖偏置、相关误差、动态到达或真实 VRAM 测量误差。

### V-G. Convergence, Runtime, and Signaling

**研究问题。** WA-MCBR 能否在有限更新、在线运行时间和模拟控制流量内完成单翻转稳定决策？

**协议。** 使用 $`N=30,50,80,100,150,200`$ 的确定性任务前缀；每个 $`N`$ 有 10 个无线实例。记录代表性目标轨迹、updates、runtime 与 signaling。QDP 仅对每个 $`N`$ 的一个实例单独报告稀疏 DP 状态数、剪枝率、峰值 live states、Python tracing peak memory 与 runtime。

**实际结果。** 代表性 WA-MCBR 轨迹只在接受翻转时严格下降，非翻转时保持不变。$`N=200`$ 时，WA-MCBR 平均 $`J=488.0714`$、updates=1708.4、runtime=92.6 ms、控制流量=201,591.2 bytes；Legacy count BR 对应 $`J=494.6753`$、updates=1136.3、runtime=9.1 ms。QDP 在 $`N=200`$ 的单独诊断为 1,991,352 个创建状态、74.6942% 剪枝、1,675 peak live states、63.0363 s runtime、0.8091 MB Python tracing peak memory。

见 [Fig. 6](05_论文图表/PNG/Fig_6_convergence_runtime_signaling.png)、[规模汇总](04_汇总表格/v_g_scale_summary.csv)、[QDP 复杂度](04_汇总表格/v_g_qdp_complexity.csv)、[代表性轨迹](03_逐运行结果/representative_convergence_trace.csv)。

**受证据约束的结论。** WA-MCBR 的公共目标单调不增，接受更新时严格下降，并在每个终态通过单翻转检查；这不是全局最优收敛证明。QDP runtime 必须与在线 runtime 分开解读。

### V-H. Failure Cases and Scope

**研究问题。** 哪些区域会暴露方法边界，而不是只展示平均结果？

**实际记录。** 在 highly constrained 主比较中，最大的 WA-MCBR 投影量化 gap 是 3.7145%，对应 $`J=276.6580`$ 和 9% 卸载率；它显示单翻转局部最优与量化对照之间仍可能存在差距。最紧联合资源条件下，hard/soft/none 的单个代表策略分别只卸载 6%/4%/7%。在 30% 且 $`q`$ 低估、$`v`$ 高估时，记录到 5.9191%-6.6373% 的单实例目标退化。远距离用户在 1.0 MB 载荷、0.5 W 下行功率或路径损耗指数 3.8 的代表性案例中出现 0% 卸载率。

见 [失败案例逐行表](03_逐运行结果/failure_cases.csv) 和 [V-H 汇总](04_汇总表格/v_h_failure_cases.csv)。

**受证据约束的结论。** 极紧资源、显著画像误差和远距离弱链路会抑制卸载或放大局部差距。该实验不能说明多小区、动态到达、真实异构 GPU 池、真实 OOM 或实时 LLM 协调下的表现。

## VI. Conclusion

本实验在共享无线链路和边缘 GPU 资源下，实现了工作量与模拟显存约束的二元卸载。基于 5 个独立任务池和 50 个无线实例，WA-MCBR 在主比较的 150 个终态中均无真实模拟资源违规并通过单翻转检查；在完整绑定网格的 1,250 个 hard 记录中同样为零违规。该证据支持其作为静态、单小区仿真中的可行局部决策方法，但不构成连续全局最优、真实 AI-RAN 部署性能或在线 LLM 可用性的证明。下一步应加入动态到达、多小区/干扰、实测无线与显存、以及端边云测试床。

## Reproduction and Audit

环境依赖见 [`01_源码与配置/requirements.txt`](01_源码与配置/requirements.txt)。本仓库已保存完成实验所需原始输入、画像、逐运行结果和图表；**验证现有工件不会调用 DeepSeek API**：

```bash
cd Mark10
bash 01_源码与配置/run_all.sh --verify-existing
```

该命令重算源数据 SHA-256、检查 500 次画像 manifest、矩阵覆盖、硬约束、图表四种格式、README 结构，并运行测试。结果写入 [acceptance audit](06_审计与复现/acceptance_audit.csv) 和 [run summary](06_审计与复现/run_summary.json)。

原始的中断绑定网格检查点只作为过程审计保留在 [`06_审计与复现`](06_审计与复现)，不参与最终汇总；最终绑定结果仅使用 [binding_resource_runs.csv](03_逐运行结果/binding_resource_runs.csv)。
