# Hotspot Imbalance Experiments

> **日期**: 2026-07-22 — 2026-07-24
> **目的**: 诊断并解决 DT-STPINN 模型对熔池高温节点（≥ 1604.85°C）完全漏判的问题

---

## 1. 背景与动机

Paper 1 的 50-epoch 模型（`logs/paper1_fast_physics_v2/best_model.pt`）在全局指标上表现尚可
（RMSE = 7.30°C, MAE = 4.05°C），但使用 `scripts/evaluate_checkpoint.py` 按温度分箱评估后
发现严重问题：**固相线以上的 367 个节点被全部漏判（Recall = 0）**。

### 1.1 数据统计

| 项目 | 数值 |
|------|------|
| VTU 文件数 | 2361（`F:/VTU/Data-00000.vtu` ~ `Data-04720.vtu`，步长 20 ms）|
| 总节点数 | 95,986 |
| 训练窗口数 | 1,641（step 0 ~ 1640）|
| 验证窗口数 | 352（step 1641 ~ 1992）|
| 测试窗口数 | 351（step 1993 ~ 2343）|

### 1.2 原始模型温度分箱评估

| 温度区间 | 节点数 | 占比 | MAE | RMSE | Bias | P99 | MaxError |
|----------|--------|------|-----|------|------|-----|----------|
| ≤ 100°C | 8,697,207 | 25.8% | 3.68 | 4.08 | −3.04 | 5.25 | 1,292 |
| 100–500°C | 4,786,654 | 14.2% | 1.67 | 3.12 | −0.51 | 4.94 | 1,172 |
| 500–1000°C | 20,200,537 | 60.0% | 4.71 | 6.57 | −4.51 | 17.62 | 953 |
| 1000–1604.85°C | 6,321 | 0.02% | 176.19 | 264.15 | −172.00 | 1,179 | 1,543 |
| 1604.85–1654.85°C (mushy) | 33 | <0.001% | 507.69 | 595.13 | −507.69 | 1,591 | 1,598 |
| ≥ 1654.85°C (liquid) | 334 | <0.001% | 716.10 | 913.03 | −716.10 | 2,255 | **2,342** |

**高温检测指标**：

| 指标 | 数值 |
|------|------|
| Recall above solidus | **0.0000** |
| Precision above solidus | 0.0000 |
| F1 above solidus | 0.0000 |
| IoU above solidus | 0.0000 |
| TP / FP / FN | 0 / 0 / 367 |

### 1.3 逐时间步最高温度

| 指标 | 真实值 | 预测值 |
|------|--------|--------|
| 温度范围 | 604.6 ~ 2948.3°C | 608.0 ~ **1536.0**°C |
| 最差步最大温差 | — | **−1444.3°C**（step 2131, t=42.62s）|

模型预测的最高温度被「压死」在 1536°C，永远达不到固相线。

### 1.4 最差节点诊断

| 属性 | 数值 |
|------|------|
| 时间步 | 2128（t ≈ 42.56 s）|
| 节点 ID | 24437 |
| 坐标 | [10.216, −9.375, 1.000] mm |
| 预测温度 | **29.38°C** |
| 真实温度 | **2370.95°C** |
| 绝对误差 | **2341.58°C** |
| 激光距离 | 7.82 mm |
| 边界标签 | −10 |
| 节点激活 | 整个窗口内始终活跃 |
| **输入窗口温度** | **20.00 / 20.00 / 20.00 / 20.00°C**（全为环境温度）|
| 一阶邻居温度 | min=20.0, max=**1886.3**, mean=1284.7°C |
| 二阶邻居温度 | min=20.0, max=1584.5, mean=922.5°C |

**核心发现**：节点在整个输入窗口内温度都是 20°C（环境温度），虽然邻居已高达 1886°C，
但 GNN 的消息传递未能将这一信号有效传导至目标节点。模型从输入中看不到任何温度即将
飙升的迹象，因此输出环境温度。

---

## 2. 基线模型配置

### 原始 50-epoch 模型（`paper1_fast_physics_v2`）

配置文件为 [`configs/paper1_fast.yaml`](../configs/paper1_fast.yaml)。

#### 模型参数

| 参数 | 值 |
|------|-----|
| hidden_dim | 64 |
| 节点特征维度 | 12 |
| 边特征维度 | 5 |
| GNN 类型 | GATv2Conv |
| GNN 层数 | 1 |
| GNN heads | 2 |
| GNN dropout | 0.1 |
| Temporal 层数 | 2 |
| Temporal heads | 4 |
| Temporal ff_dim | 256 |
| Fusion 类型 | cross_attention |
| Fusion heads | 4 |
| 解码器 | 仅温度头 |

#### 材料参数（Ti-6Al-4V）

| 参数 | 值 |
|------|-----|
| 密度 ρ | 4430.0 kg/m³ |
| 比热 Cp | 526.3 J/(kg·K) |
| 热导率 k | 6.7 W/(m·K) |
| 固相线 | 1604.85°C |
| 液相线 | 1654.85°C |
| 辐射率 | 0.35 |
| 对流系数 | 10.0 W/(m²·K) |
| 环境温度 | 20.0°C |

#### 物理约束

| 约束 | 启用 |
|------|------|
| heat_conduction (PDE) | ✅ |
| boundary_convection | ✅ |
| boundary_radiation | ❌ |
| initial_condition | ✅ |
| fourier_flux | ❌ |

#### Loss 权重

| 项 | λ |
|----|-----|
| λ_T (数据) | 1.0 |
| λ_PDE | 0.1 |
| λ_BC | 0.1 |
| λ_IC | 0.5 |
| λ_smooth | 0.01 |
| λ_hot (高温) | 0.0（未启用）|

#### 训练配置

| 参数 | 值 |
|------|-----|
| Batch size | 1 |
| Gradient accumulation | 4 |
| Epochs | 50 |
| Learning rate | 1e-3 |
| Scheduler | CosineAnnealingLR (eta_min=1e-6) |
| Weight decay | 1e-5 |
| Grad clip | 1.0 |
| AMP | bfloat16 |
| Window size | 4 |
| k-NN | 16 |
| Mesh edges | ✅ |

#### 结果

| 指标 | 数值 |
|------|------|
| Best val loss | — |
| Test RMSE | 7.30 |
| Test MAE | 4.05 |
| R² | — |
| Max error | 2341.58 |
| Recall above solidus | **0.00** |

---

## 3. 消融实验

为验证高温类不平衡问题的根本原因，进行了三组 15-epoch 对照实验。
三组使用完全相同的模型架构和训练超参数，仅改变 loss 和采样策略。

### 3.1 共同配置（三组一致）

除实验变量外，以下参数与 `paper1_fast.yaml` 一致，仅 epochs 从 50 降为 15：

| 参数 | 值 |
|------|-----|
| hidden_dim | 64 |
| GNN 层数 | 1（heads=2）|
| Temporal 层数 | 2（heads=4, ff_dim=256）|
| window_size | 4 |
| accumulate_grad_batches | 4 |
| lr | 1e-3 |
| epochs | **15** |

### 3.2 实验 A：基线（Baseline）

**配置文件**：[`configs/ablation_a_baseline.yaml`](../configs/ablation_a_baseline.yaml)

| 变量 | 值 |
|------|-----|
| λ_hot | **0.0**（关闭）|
| stratified_sampling | **false** |

**训练输出**：

```
Best validation loss: 320.32 at epoch 15
Test metrics:
  val_loss: 113.11   RMSE: 9.95   MAE: 4.40   R²: 0.9988
  MaxError: 2442.34  P50: 3.45  P90: 9.24  P95: 11.45  P99: 16.81
Worst node: step 2131, node 24219, [10.417, -0.521, 1.000] mm
  Prediction: 506°C   Target: 2948°C   Error: 2442°C
```

### 3.3 实验 B：高温加权 Loss

**配置文件**：[`configs/ablation_b_hotloss.yaml`](../configs/ablation_b_hotloss.yaml)

**策略**：在 MSE loss 上叠加温度相关权重。
$$
w_i = 1.0 + \lambda_{\text{hot\_weight}} \cdot \left(\frac{\max(0,\ T_{\text{target},i} - T_{\text{threshold}})}{T_{\text{ref}} - T_{\text{threshold}}}\right)^{p}
$$
其中 $T_{\text{threshold}}=500$°C, $T_{\text{ref}}=1654.85$°C（液相线）, $p=2$, $\lambda_{\text{hot\_weight}}=5$.

最终 loss：$L = L_{\text{T}} + \lambda_{\text{hot}} \cdot L_{\text{hotspot}} + \dots$

| 变量 | 值 |
|------|-----|
| λ_hot | **1.0** |
| hot_threshold | **500.0** |
| hot_weight | **5.0** |
| hot_weight_power | **2.0** |
| stratified_sampling | **false** |

**训练输出**：

```
Best validation loss: 748.44 at epoch 15
Test metrics:
  val_loss: 340.80   RMSE: 9.82   MAE: 3.86   R²: 0.9988
  MaxError: 2333.20  P50: 2.85  P90: 8.42  P95: 10.81  P99: 16.75
Worst node: step 2128, node 24437, [10.216, -9.375, 1.000] mm
  Prediction: 38°C   Target: 2371°C   Error: 2333°C
```

### 3.4 实验 C：高温加权 Loss + 分层窗口采样

**配置文件**：[`configs/ablation_c_hotsample.yaml`](../configs/ablation_c_hotsample.yaml)

**策略**：在实验 B 的 loss 基础上，训练时按窗口的最大目标温度分三类采样：

| 类别 | 阈值 | 采样比例 |
|------|------|----------|
| normal | max T < 500°C | 50% |
| hot | 500°C ≤ max T < 1604.85°C | 30% |
| melting | max T ≥ 1604.85°C | 20% |

注意：验证集和测试集**不受影响**，保持原始时间顺序。

| 变量 | 值 |
|------|-----|
| λ_hot | **1.0** |
| hot_threshold | **500.0** |
| hot_weight | **5.0** |
| hot_weight_power | **2.0** |
| stratified_sampling | **true** |
| stratified 比例 | 50% / 30% / 20% |

**训练输出**：

```
Best validation loss: 749.19 at epoch 15
Test metrics:
  val_loss: 345.03   RMSE: 9.92   MAE: 4.00   R²: 0.9988
  MaxError: 2334.95  P50: 3.10  P90: 8.33  P95: 10.54  P99: 16.39
Worst node: step 2128, node 24437, [10.216, -9.375, 1.000] mm
  Prediction: 36°C   Target: 2371°C   Error: 2335°C
```

---

## 4. 结果对比与分析

### 4.1 全局指标

| 指标 | 原始 50-epoch | A (基线) | B (hotspot loss) | C (loss + 采样) |
|------|:---:|:---:|:---:|:---:|
| Epochs | 50 | 15 | 15 | 15 |
| Val Loss (best) | — | 320.3 | 748.4 | 749.2 |
| Test val_loss | — | 113.1 | 340.8 | 345.0 |
| **RMSE** | **7.30** | 9.95 | **9.82** | 9.92 |
| **MAE** | 4.05 | 4.40 | **3.86** | 4.00 |
| **P50** | 3.66 | 3.45 | **2.85** | 3.10 |
| **P90** | **7.42** | 9.24 | 8.42 | **8.33** |
| **P95** | **8.90** | 11.45 | 10.81 | **10.54** |
| **P99** | **14.37** | 16.81 | 16.75 | **16.39** |
| **MaxError** | **2342** | 2442 | **2333** | 2335 |
| R² | — | 0.9988 | 0.9988 | 0.9988 |

> 注：原始 50-epoch 模型训练更充分（50 vs 15 epochs），因此全局指标普遍优于消融组
> 是正常现象。核心关注点在高温区的改善幅度。

### 4.2 最差节点预测值

| 实验 | 预测 | 目标 | 误差 | 节点 |
|------|------|------|------|------|
| 原始 50-epoch | **29.4°C** | 2371°C | 2342°C | 24437 |
| A (baseline) | **506.0°C** | 2948°C | 2442°C | 24219 |
| B (hotspot loss) | **37.8°C** | 2371°C | 2333°C | 24437 |
| C (loss+sampling) | **36.0°C** | 2371°C | 2335°C | 24437 |

### 4.3 关键发现

1. **Hotspot loss 对最差节点几乎无效**：B 和 C 的最差节点预测值（38°C / 36°C）与
   原始模型（29°C）处于同一量级——都是环境温度级别。MaxError 仅从 2342 微降到 2333（−0.4%）。

2. **分层采样没有额外增益**：C 在 B 的基础上增加了 20% 熔化窗口采样，但几乎所有指标
   都与 B 持平或略差，最差节点预测几乎相同。

3. **中低温区有轻微改善**：B 和 C 的 P50 从 3.45 降到 2.85/3.10，说明 hotspot loss
   确实让模型对中高温区（500–1000°C）更加关注。但这属于「让已经好的区域更好」，
   没有解决核心问题。

4. **B/C 的 val_loss 远高于 A**（748 vs 320）：这不是模型变差，而是 hotspot loss 项
   自身贡献了巨大数值——高温节点的 sq_error 本身就很大，额外权重放大了这一点。

5. **A 组的最差节点换了一个**（24219 而非 24437），且预测值 506°C 明显高于 B/C 的
   36-38°C。这暗示 B/C 的 hotspot loss 可能在个别节点上产生了**反向效果**——过度
   惩罚了某几个高温节点，反而让模型对它们更保守。

### 4.4 根本原因

Hotspot loss 和 stratified sampling 无效的深层原因是**输入信息不足以预测温度突变**。

最差节点 24437 的诊断数据明确显示：
- 输入窗口的 4 个时间步温度全为 **20°C**（环境温度）
- 一阶邻居最高已达 **1886°C**，但 GATv2Conv 的消息传递未能将这一信号传导至目标节点
- 模型看到的输入序列是 `[20, 20, 20, 20]`，没有任何升温趋势

在这种情况下，无论 loss 权重多大，模型都无法从纯环境温度的输入推断出 2370°C 的输出。
问题出在**特征表示和信息流**，而非 loss 函数。

---

## 5. 相关工作

### 5.1 改动清单

为支持消融实验，对代码库做了以下修改：

| 文件 | 改动 |
|------|------|
| [`src/config.py`](../src/config.py) | `LossConfig` 新增 `lambda_hot`, `hot_threshold`, `hot_weight`, `hot_weight_power`；`DataConfig` 新增 `stratified_sampling` 及比例参数 |
| [`src/loss.py`](../src/loss.py) | `DTSTPINNLoss.forward()` 新增 hotspot 加权项 |
| [`src/data/dataset.py`](../src/data/dataset.py) | 新增 `StratifiedWindowSampler` 类 |
| [`scripts/train.py`](../scripts/train.py) | 集成 stratified sampler（仅训练集）|
| [`scripts/evaluate_checkpoint.py`](../scripts/evaluate_checkpoint.py) | **新增**：温度分箱评估、高温检测指标、最差节点诊断、VTU 导出 |

### 5.2 新增配置文件

| 文件 | 用途 |
|------|------|
| [`configs/ablation_a_baseline.yaml`](../configs/ablation_a_baseline.yaml) | 15-epoch 基线 |
| [`configs/ablation_b_hotloss.yaml`](../configs/ablation_b_hotloss.yaml) | 仅 hotspot loss |
| [`configs/ablation_c_hotsample.yaml`](../configs/ablation_c_hotsample.yaml) | hotspot loss + 分层采样 |

---

## 6. 下一步方向

Hotspot loss 和采样策略已被验证无效，问题根因在**输入表示**而非损失函数。
后续可行的方向：

| 优先级 | 方向 | 说明 |
|:------:|------|------|
| **1** | **增强节点特征** | 将激光距离、激光功率、到激光的相对方向等作为显式节点特征注入，让模型在输入窗口内就能感知「激光正在接近」 |
| **2** | **增大时间窗口** | window_size 从 4 → 16 或更大，让输入覆盖温度从环境到熔化的完整上升过程 |
| **3** | **多步预测** | 预测未来多步（predict_steps > 1），迫使模型学习温度变化的动力学而非静态映射 |
| **4** | **图结构改进** | 增加跳跃连接或更深的 GNN，改善热点信息在邻居间的传播效率 |
| **5** | **双向 GNN + 边特征增强** | 让边特征编码温度梯度方向，让 GNN 显式感知「热量正在向这个节点传播」|
| **6** | **物理引导的特征** | 基于热方程计算每个节点的预期温升速率，作为额外输入特征，让模型有物理先验 |

**重申**：在上述任一方向取得验证性改善之前，**不要扩大模型容量**（hidden_dim, GNN 层数），
否则只会更精确地拟合低温节点而继续忽略熔池。

---

*最后更新：2026-07-24*
