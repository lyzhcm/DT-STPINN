# DT-STPINN: Architecture Document

> Note: this legacy document has encoding damage in the current checkout.
> The maintained Paper 1 architecture that matches the current code is
> `docs/ARCHITECTURE_PAPER1.md`.

> **Dynamic Twin Spatio-Temporal Physics-informed Neural Network**
>
> *面向 DED 薄壁件热-力耦合数字孪生的动态图时空物理信息网络*

---

## 目录

1. [项目定位](#1-项目定位)
2. [系统总体架构](#2-系统总体架构)
3. [模块一：Dynamic Graph Builder](#3-模块一dynamic-graph-builder)
4. [模块二：Spatial Encoder](#4-模块二spatial-encoder)
5. [模块三：Temporal Encoder](#5-模块三temporal-encoder)
6. [模块四：Cross Fusion Module](#6-模块四cross-fusion-module)
7. [模块五：Thermo-Mechanical Decoder](#7-模块五thermo-mechanical-decoder)
8. [模块六：Physics Constraint Module](#8-模块六physics-constraint-module)
9. [模块七：Digital Twin Engine](#9-模块七digital-twin-engine)
10. [模块八：MPC Optimization](#10-模块八mpc-optimization)
11. [Loss 设计](#11-loss-设计)
12. [技术栈](#12-技术栈)
13. [论文路线图](#13-论文路线图)
14. [目录结构](#14-目录结构)

---

## 1. 项目定位

### 1.1 模型名称

- **主名称**: DT-STPINN (Dynamic Twin Spatio-Temporal Physics-informed Neural Network)
- **学术别名**: DGTFormer (Dynamic Graph Thermo-mechanical Transformer)

### 1.2 论文目标

> 利用动态图表示打印几何演化，利用时空 Transformer 学习热历史，利用 PINN 保证热力学一致性，实现薄壁件实时热-力状态预测及工艺优化。

### 1.3 目标期刊

- *Additive Manufacturing*
- *Computer Methods in Applied Mechanics and Engineering (CMAME)*
- *Engineering Applications of Artificial Intelligence*
- *Journal of Computational Physics*

---

## 2. 系统总体架构

系统共分为 **8 个核心模块**，形成完整闭环：

```text
                            ┌─────────────────────┐
                            │     CAD Model        │
                            └──────────┬──────────┘
                                       │
                                       ▼
                   ┌───────────────────────────────────────┐
                   │     ① Dynamic Graph Builder           │
                   │     动态图构建: G_t = (V_t, E_t)       │
                   └───────────────────┬───────────────────┘
                                       │ 动态图 G_t
                                       ▼
                   ┌───────────────────────────────────────┐
                   │     ② Spatial Encoder                 │
                   │     GNN (GATv2 + Edge Feature)        │
                   └───────────────────┬───────────────────┘
                                       │ 空间特征 Z_s ∈ R^(N×256)
                                       ▼
                   ┌───────────────────────────────────────┐
                   │     ③ Temporal Encoder                │
                   │     Transformer (4层 / 8头 / 256维)   │
                   └───────────────────┬───────────────────┘
                                       │ 时序特征 Z_t ∈ R^(N×256)
                                       ▼
                   ┌───────────────────────────────────────┐
                   │     ④ Cross Fusion Module             │
                   │     Cross-Attention (Spatial→Temporal) │
                   └───────────────────┬───────────────────┘
                                       │ 融合特征 Z_f ∈ R^(N×256)
                                       ▼
                   ┌───────────────────────────────────────┐
                   │     ⑤ Thermo-Mechanical Decoder       │
                   │     多任务解码 (4 Heads)               │
                   └──┬─────────┬─────────┬────────────────┘
                      │         │         │
                      ▼         ▼         ▼
                   ┌────┐  ┌────┐   ┌────┐
                   │ T  │  │ σ  │   │ u  │   (温度/应力/位移/热流)
                   └────┘  └────┘   └────┘
                      │         │         │
                      └─────────┼─────────┘
                                ▼
                   ┌───────────────────────────────────────┐
                   │     ⑥ Physics Constraint Module       │
                   │     PDE Loss (热/力/能量/边界)         │
                   └───────────────────┬───────────────────┘
                                       │ Loss → 反向传播
                                       ▼
                   ┌───────────────────────────────────────┐
                   │     ⑦ Digital Twin Engine             │
                   │     实时推理 + 未来状态预测            │
                   └───────────────────┬───────────────────┘
                                       │
                                       ▼
                   ┌───────────────────────────────────────┐
                   │     ⑧ MPC Optimization               │
                   │     工艺参数在线优化                   │
                   └───────────────────────────────────────┘
```

---

## 3. 模块一：Dynamic Graph Builder

### 3.1 设计动机

DED（定向能量沉积）打印过程中，几何形状随逐层沉积而演化。传统 GNN 假设图结构固定，无法处理动态增材过程。

### 3.2 动态图定义

$$
G_t = (V_t, E_t)
$$

每一层打印完成后：
- **新增节点**: 新建沉积层内的离散化节点
- **新增边**: 新建层内及层间连接
- **更新边权**: 随温度/应力场变化更新导热率/刚度等物理属性

### 3.3 Node Feature 设计

每个节点 $v_i$ 的特征维度为 **18–24 维**，结构如下：

| 类别 | 特征 | 符号 | 维度 |
|------|------|------|------|
| **几何** | 坐标 x, y, z | $\mathbf{p}$ | 3 |
| **热学** | 温度 | $T$ | 1 |
|  | 冷却速率 | $\dot{T}$ | 1 |
|  | 温度梯度 | $\nabla T$ | 3 |
| **力学** | 应力 (6 分量) | $\sigma_{ij}$ | 6 |
|  | 应变 (6 分量) | $\varepsilon_{ij}$ | 6 |
| **工艺** | 层号 | $n_{\ell}$ | 1 |
|  | 到激光距离 | $d_{\text{laser}}$ | 1 |
|  | 扫描方向 | $\theta_{\text{scan}}$ | 1 |
|  | 打印时间 | $t_{\text{print}}$ | 1 |
| **材料** | 密度 | $\rho$ | 1 |
|  | 比热容 | $C_p$ | 1 |
|  | 热导率 | $k$ | 1 |
|  | 热膨胀系数 | $\alpha$ | 1 |
|  | 弹性模量 | $E$ | 1 |
|  | 泊松比 | $\nu$ | 1 |

> **注**: 训练初期力学特征可来自 FEA 数据，预测阶段由模型自回归生成。

### 3.4 Edge Feature 设计

边特征 $e_{ij}$ 不仅包含几何距离，还编码物理语义：

| 特征 | 符号 | 说明 |
|------|------|------|
| 欧氏距离 | $d_{ij}$ | 节点间距 |
| 方向向量 | $\Delta x, \Delta y, \Delta z$ | 相对位置 |
| 平均热导率 | $k_{ij}$ | $k_{ij} = (k_i + k_j)/2$ |
| 夹角 | $\theta_{ij}$ | 与热流/力方向的夹角 |
| 壁厚 | $\tau_{ij}$ | 沿薄壁方向的厚度 |

### 3.5 注意力机制设计

采用 **Physics-aware Graph Attention**：

$$
\alpha_{ij} = \text{softmax}\big(\mathsf{a}^\top \cdot \text{LeakyReLU}(\mathbf{W}[x_i \| x_j \| e_{ij}])\big)
$$

即 Attention 不仅考虑节点特征 $x_i, x_j$，还显式编码边特征 $e_{ij}$。

---

## 4. 模块二：Spatial Encoder

### 4.1 设计选择

- **基准模型**: GATv2Conv（支持 Edge Feature）
- **不使用 GCN**: Attention 机制更适合非均匀热扩散场景

### 4.2 网络结构

```
Input: Node Feature X ∈ R^(N×D_n), Edge Feature E ∈ R^(M×D_e), Adjacency A
                    │
    ┌───────────────┼───────────────┐
    │  Linear(D_n → 256)           │
    │  LayerNorm                    │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┼───────────────┐
    │  GATv2Conv(256 → 256, heads=8) │
    │  BatchNorm                     │
    │  GELU                          │
    │  Residual                      │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┼───────────────┐
    │  GATv2Conv(256 → 256, heads=8) │
    │  BatchNorm                     │
    │  GELU                          │
    │  Residual                      │
    └───────────────┬───────────────┘
                    │
            Output: Z_s ∈ R^(N×256)
```

### 4.3 关键参数

| 参数 | 值 |
|------|-----|
| 输入维度 | 18–24 (Node Feature) |
| 隐藏维度 | 256 |
| 注意力头数 | 8 |
| GAT 层数 | 2 |
| 激活函数 | GELU |
| 归一化 | LayerNorm + BatchNorm |
| 残差连接 | 每层 |

---

## 5. 模块三：Temporal Encoder

### 5.1 设计动机

核心创新点：Transformer **不是**学习简单的时间序列，而是学习**完整热历史**（thermal history）。热历史对于残余应力累积和变形预测至关重要。

### 5.2 输入 Token 设计

每个时间步的 Token 包含多模态信息：

| 特征 | 说明 |
|------|------|
| Temperature | 当前温度场 |
| Laser Position | 激光位置 |
| Power | 激光功率 |
| Speed | 扫描速度 |
| Layer Index | 当前层号 |
| Scan Direction | 扫描方向 (one-hot / embedding) |
| Cooling Time | 已冷却时长 |

### 5.3 时间采样策略

采用**对数采样**（log-spaced sampling）:

$$
\text{sampling times} \in \{1, 2, 4, 8, 16, 32, 64, \dots\}
$$

**理由**: 热累积过程时间跨度大（毫秒级激光加热 ~ 秒级层间冷却），对数采样能同时捕获短时瞬态和长时稳态特征。

### 5.4 Transformer 结构

```
Input: Temporal Tokens ∈ R^(B×T×D)
                    │
    ┌───────────────┼───────────────┐
    │  Positional Encoding         │
    │  (Sinusoidal / Learnable)    │
    └───────────────┬───────────────┘
                    │
    ┌───────────────┼───────────────┐
    │  TransformerEncoder × 4      │
    │  │  MultiHeadAttention(8头)  │
    │  │  LayerNorm                 │
    │  │  FFN (256→1024→256)        │
    │  │  LayerNorm                 │
    │  │  Residual                  │
    └───────────────┬───────────────┘
                    │
            Output: Z_t ∈ R^(N×256)
```

### 5.5 关键参数

| 参数 | 值 |
|------|-----|
| Token 维度 | 256 |
| 层数 | 4 |
| 注意力头数 | 8 |
| FFN 隐层维度 | 1024 |
| 激活函数 | GELU |
| Dropout | 0.1 |
| 位置编码 | Sinusoidal |

---

## 6. 模块四：Cross Fusion Module

### 6.1 设计选择

**不使用简单 Concat**。采用 **Cross-Attention** 机制，使空间特征主动查询时序特征：

```
         Spatial Feature Z_s          Temporal Feature Z_t
                    │                         │
                    ▼                         ▼
              ┌──────────┐              ┌──────────┐
              │  Linear  │              │  Linear  │
              │  (Query) │              │  (Key +  │
              │  Q       │              │   Value) │
              └────┬─────┘              └────┬─────┘
                   │                         │
                   └──────────┬──────────────┘
                              │
                              ▼
                ┌─────────────────────────┐
                │   Multi-Head Cross-     │
                │   Attention             │
                │                         │
                │   Attn(Q_s, K_t, V_t)   │
                └─────────────┬───────────┘
                              │
                              ▼
                ┌─────────────────────────┐
                │   Feed-Forward + Norm   │
                └─────────────┬───────────┘
                              │
                              ▼
                     Z_f ∈ R^(N×256)
```

### 6.2 物理直觉

- **Query** 来自空间特征: "我这个空间位置，历史中哪些时刻最重要？"
- **Key / Value** 来自时序特征: 提供热历史信息
- Transformer 学会主动关注影响最大的历史时段

---

## 7. 模块五：Thermo-Mechanical Decoder

### 7.1 多任务输出设计

融合特征 $Z_f$ 后，使用 **4 个独立 Head** 进行多任务解码：

```
                        Shared Feature Z_f
                    │
    ┌───────────────┼───────────────────────────┐
    │               │              │            │
    ▼               ▼              ▼            ▼
┌─────────┐   ┌─────────┐   ┌─────────┐  ┌──────────┐
│  Temp   │   │  Stress │   │  Disp   │  │   Heat   │
│  Head   │   │  Head   │   │  Head   │  │  Flux    │
│         │   │         │   │         │  │  Head    │
│ MLP(2层)│   │ MLP(2层)│   │ MLP(2层)│  │ MLP(2层) │
│ 256→128 │   │ 256→128 │   │ 256→128 │  │ 256→128  │
│   →1    │   │  →6     │   │  →3     │  │   →3     │
└────┬────┘   └────┬────┘   └────┬────┘  └────┬─────┘
     │             │             │            │
     ▼             ▼             ▼            ▼
   T̂ ∈ ℝ       σ̂ ∈ ℝ^6     û ∈ ℝ^3      q̂ ∈ ℝ^3
 (温度)      (应力张量)    (位移)      (热流)
```

### 7.2 各 Head 说明

| Head | 输出 | 维度 | 用途 |
|------|------|------|------|
| Temperature Head | $\hat{T}$ | 1 | 温度场预测 |
| Stress Head | $\hat{\sigma}_{ij}$ | 6 | 残余应力张量 (Voigt 记号) |
| Displacement Head | $\hat{u}_i$ | 3 | 位移场 |
| Heat Flux Head | $\hat{q}_i$ | 3 | 热流通量 (Physics Loss 输入) |

> **Heat Flux Head 的重要性**: 热流 $q$ 是热传导 PDE 的关键变量，显式输出使 Physics Loss 能够直接检验 Fourier 定律。

---

## 8. 模块六：Physics Constraint Module

### 8.1 约束方程汇总

| 编号 | 物理方程 | 公式 |
|------|----------|------|
| PDE-1 | 瞬态热传导 | $\rho C_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + Q_{\text{laser}}$ |
| PDE-2 | Fourier 热流 | $\mathbf{q} = -k \nabla T$ |
| PDE-3 | 热应变 | $\varepsilon_{th} = \alpha (T - T_0) \mathbf{I}$ |
| PDE-4 | 几何方程 | $\varepsilon = \frac{1}{2}(\nabla u + \nabla u^T)$ |
| PDE-5 | 线弹性本构 | $\sigma = \mathbf{D} : (\varepsilon - \varepsilon_{th})$ |
| PDE-6 | 力学平衡 | $\nabla \cdot \sigma = 0$ |
| PDE-7 | 能量守恒 | $Q_{\text{laser}} = Q_{\text{cond}} + Q_{\text{conv}} + Q_{\text{rad}} + Q_{\text{store}}$ |

### 8.2 边界条件

| 边界类型 | 条件 | 数学表达 |
|----------|------|----------|
| 底板固定 | Dirichlet | $u = 0$ on $\Gamma_{\text{base}}$ |
| 侧面对流 | Robin | $k \frac{\partial T}{\partial n} = h (T - T_{\infty})$ on $\Gamma_{\text{side}}$ |
| 顶面辐射 | Nonlinear Robin | $k \frac{\partial T}{\partial n} = \epsilon \sigma_{SB} (T^4 - T_{\infty}^4)$ on $\Gamma_{\text{top}}$ |

### 8.3 自动微分实现

利用 **PyTorch Autograd** 计算空间/时间导数：

```python
# 空间梯度: ∇T = ∂T/∂x, ∂T/∂y, ∂T/∂z
dT_dx = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T),
                             create_graph=True, retain_graph=True)[0]

# 热传导残差
R_heat = rho * Cp * dT_dt - div(k * grad_T) - Q_laser
```

---

## 9. 模块七：Digital Twin Engine

### 9.1 推理流程

```text
    ┌──────────┐
    │  Sensor  │  (IR相机 / 热电偶 / 激光位移)
    └────┬─────┘
         │ 实时数据
         ▼
    ┌──────────────┐
    │  Graph Update │  更新节点温度和工艺状态
    └──────┬───────┘
         │
         ▼
    ┌──────────────┐
    │   Inference   │  DT-STPINN 前向推理
    └──────┬───────┘
         │
         ▼
    ┌─────────────────────────────┐
    │  Autoregressive Prediction  │
    │  预测未来 K 步状态:         │
    │  t+1, t+2, ..., t+K        │
    └─────────────┬───────────────┘
         │
         ▼
    ┌─────────────────────────────┐
    │  Output:                    │
    │  • Temperature Field        │
    │  • Residual Stress Field   │
    │  • Deformation (Warp)      │
    └─────────────────────────────┘
```

### 9.2 预测时间窗

| 时间 | 用途 |
|------|------|
| +2 s | 在线实时监控 |
| +5 s | 短期预警 |
| +10 s | 工艺调整窗口 |
| +20 s | 层间干预 |

---

## 10. 模块八：MPC Optimization

### 10.1 优化框架

基于 DT-STPINN 的快速推理能力，构建在线工艺优化回路：

```text
    ┌─────────────────────┐
    │  当前状态 X_t       │
    └──────────┬──────────┘
               │
    ┌──────────┼──────────────────────┐
    │  Parameter Sweep                │
    │  ┌─────────────────────────┐    │
    │  │ P = {850, 900, 950} W   │───▶│  DT-STPINN 并行推理
    │  │ v = {8, 10, 12} mm/s   │    │
    │  │ path = {A, B, C}       │    │
    │  └─────────────────────────┘    │
    └──────────────────┬──────────────┘
                       │
                       ▼
    ┌──────────────────────────────┐
    │  Evaluate Objective:         │
    │  min ‖u‖_2 (最小化变形)      │
    │  s.t. T_max ≤ T_limit       │
    │       σ_max ≤ σ_yield       │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │  Optimal Parameters:         │
    │  (P*, v*, path*)           │
    └──────────────────────────────┘
```

### 10.2 优化变量与约束

| 类别 | 变量 | 约束 |
|------|------|------|
| 控制变量 | 功率 $P$ | $P_{\min} \leq P \leq P_{\max}$ |
| 控制变量 | 速度 $v$ | $v_{\min} \leq v \leq v_{\max}$ |
| 控制变量 | 路径 | 离散选择 |
| 约束 | 最高温度 | $T \leq T_{\text{limit}}$ |
| 约束 | 最大应力 | $\sigma \leq \sigma_{\text{yield}}$ |
| 目标 | 最小变形 | $\min \|u\|_2$ |

---

## 11. Loss 设计

### 11.1 总损失函数

$$
\mathcal{L} = \lambda_1 L_T + \lambda_2 L_{\text{PDE}} + \lambda_3 L_\sigma + \lambda_4 L_u + \lambda_5 L_{\text{Energy}} + \lambda_6 L_{\text{BC}} + \lambda_7 L_{\text{Smooth}}
$$

### 11.2 各分量说明

| 符号 | 名称 | 公式 | 权重 | 说明 |
|------|------|------|------|------|
| $L_T$ | 温度监督 | $\|T - \hat{T}\|^2$ | $\lambda_1=1.0$ | 数据驱动 |
| $L_{\text{PDE}}$ | 热传导残差 | $\|\rho C_p \dot{T} - \nabla\cdot(k\nabla T) - Q\|^2$ | $\lambda_2=0.1$ | 物理约束 |
| $L_\sigma$ | 应力监督 | $\|\sigma - \hat{\sigma}\|^2$ | $\lambda_3=0.5$ | 数据+物理 |
| $L_u$ | 位移监督 | $\|u - \hat{u}\|^2$ | $\lambda_4=0.5$ | 数据+物理 |
| $L_{\text{Energy}}$ | 能量守恒 | $\|Q_{\text{in}} - Q_{\text{out}}\|^2$ | $\lambda_5=0.05$ | 全局物理 |
| $L_{\text{BC}}$ | 边界约束 | Dirichlet/Robin Residual | $\lambda_6=0.1$ | 边界物理 |
| $L_{\text{Smooth}}$ | 图平滑 | $\sum_{ij} A_{ij}\|z_i - z_j\|^2$ | $\lambda_7=0.01$ | 正则化 |

### 11.3 权重调度策略

推荐使用 **自适应权重** (如 *Learning to Reweight* 或 *GradNorm*)，而非手动固定。

---

## 12. 技术栈

| 模块 | 技术选型 | 备注 |
|------|----------|------|
| 图构建 | **PyTorch Geometric (PyG)** | GATv2Conv 原生支持 Edge Feature |
| GNN | **GATv2Conv** + 可扩展 Edge Attr | `GATv2Conv(edge_dim=D_e)` |
| Transformer | **PyTorch TransformerEncoder** | 或 FlashAttention 加速 |
| PINN 微分 | **PyTorch Autograd** | `torch.autograd.grad` |
| FEA 数据集 | **ANSYS / Abaqus / Simufact Additive** | 监督数据来源 |
| 数据存储 | **HDF5** | 按层和时间存储图数据 |
| 可视化 | **TensorBoard** / **Weights & Biases** | 训练监控 |
| 场可视化 | **ParaView** / **VTK** | 温度/应力场渲染 |
| MPC 优化 | **CasADi** 或 贝叶斯优化 (gpytorch/Botorch) | 在线优化 |

---

## 13. 论文路线图

建议分 **三篇论文** 完成，每篇具有独立创新点，共享同一架构基础：

```text
┌─────────────────────────────────────────────────────────┐
│                    总体架构 (本文档)                      │
└──────────────────────────┬──────────────────────────────┘
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
    ▼                      ▼                      ▼
┌────────┐          ┌────────────┐          ┌────────────┐
│Paper 1 │          │  Paper 2   │          │  Paper 3   │
│        │          │            │          │            │
│算法基础 │  ──────▶ │ 热-力耦合  │  ──────▶ │ 数字孪生   │
│        │          │            │          │            │
│温度预测 │          │ 多任务预测  │          │ 在线优化   │
└────────┘          └────────────┘          └────────────┘
```

### Paper 1: 算法基础 (温度场预测)

- **创新点**: 动态图构建 + 时空 PINN 实现薄壁件温度场预测
- **核心模块**: ①②③⑥⑨
- **数据**: 单物理场 FEA 温度数据
- **投稿思路**: *Engineering Applications of AI* / *Computational Materials Science*

### Paper 2: 热-力耦合 (多任务扩展)

- **创新点**: 多任务解码器 + 热-力联合 Physics Loss
- **核心模块**: ①–⑥
- **数据**: 完整热-力耦合 FEA 数据 (T, σ, u)
- **投稿思路**: *CMAME* / *Additive Manufacturing*

### Paper 3: 数字孪生系统 (工程应用)

- **创新点**: 在线推理引擎 + MPC 工艺优化闭环
- **核心模块**: ①–⑧
- **数据**: 实际打印实验数据 + 传感器融合
- **投稿思路**: *Additive Manufacturing* / *Journal of Manufacturing Processes*

---

## 14. 目录结构

```
DT-STPINN/
├── docs/
│   └── ARCHITECTURE.md          # 本文档
├── data/
│   ├── raw/                     # 原始 FEA 数据
│   ├── processed/               # 预处理后的 HDF5 图数据
│   └── configs/                 # 数据生成配置文件
├── src/
│   ├── graph_builder/           # 模块①: Dynamic Graph Builder
│   │   ├── __init__.py
│   │   ├── node_features.py
│   │   ├── edge_features.py
│   │   └── dynamic_graph.py
│   ├── encoder/                 # 模块②③④
│   │   ├── __init__.py
│   │   ├── spatial_encoder.py   # GATv2 空间编码器
│   │   ├── temporal_encoder.py  # Transformer 时序编码器
│   │   └── cross_fusion.py      # Cross-Attention 融合
│   ├── decoder/                 # 模块⑤: Thermo-Mechanical Decoder
│   │   ├── __init__.py
│   │   ├── heads.py             # 4 个预测 Head
│   │   └── decoder.py
│   ├── physics/                 # 模块⑥: Physics Constraint
│   │   ├── __init__.py
│   │   ├── pde_losses.py        # PDE 残差计算
│   │   ├── boundary.py          # 边界条件
│   │   └── energy.py            # 能量守恒约束
│   ├── engine/                  # 模块⑦: Digital Twin Engine
│   │   ├── __init__.py
│   │   ├── inferencer.py        # 推理引擎
│   │   └── autoregressive.py    # 自回归预测
│   ├── mpc/                     # 模块⑧: MPC Optimization
│   │   ├── __init__.py
│   │   ├── optimizer.py         # 优化器
│   │   └── parametrization.py   # 参数化
│   ├── loss.py                  # 总损失函数
│   ├── model.py                 # 完整 DT-STPINN 模型
│   └── trainer.py               # 训练循环
├── scripts/
│   ├── preprocess.py            # 数据预处理
│   ├── train.py                 # 训练入口
│   ├── evaluate.py              # 评估
│   └── demo_twin.py             # 数字孪生演示
├── notebooks/
│   └── exploration.ipynb        # 探索性分析
├── configs/
│   ├── default.yaml             # 默认配置
│   └── experiment/              # 实验配置
├── tests/
│   └── ...
├── requirements.txt
├── setup.py
└── README.md
```

---

## 附录 A: 符号表

| 符号 | 含义 | 单位 |
|------|------|------|
| $T$ | 温度 | K |
| $\rho$ | 密度 | kg/m³ |
| $C_p$ | 比热容 | J/(kg·K) |
| $k$ | 热导率 | W/(m·K) |
| $\alpha$ | 热膨胀系数 | 1/K |
| $E$ | 弹性模量 | Pa |
| $\nu$ | 泊松比 | — |
| $\sigma$ | 应力 | Pa |
| $\varepsilon$ | 应变 | — |
| $u$ | 位移 | m |
| $q$ | 热流 | W/m² |
| $Q$ | 热源 | W/m³ |
| $h$ | 对流换热系数 | W/(m²·K) |
| $\epsilon$ | 辐射率 | — |
| $\sigma_{SB}$ | Stefan-Boltzmann 常数 | W/(m²·K⁴) |
| $P$ | 激光功率 | W |
| $v$ | 扫描速度 | m/s |

## 附录 B: 维度速查

| 张量 | 维度 |
|------|------|
| Node Feature $\mathbf{X}_t$ | $\mathbb{R}^{N_t \times 24}$ |
| Edge Feature $\mathbf{E}_t$ | $\mathbb{R}^{M_t \times D_e}$ |
| Spatial Feature $\mathbf{Z}_s$ | $\mathbb{R}^{N \times 256}$ |
| Temporal Token | $\mathbb{R}^{B \times T \times 256}$ |
| Fused Feature $\mathbf{Z}_f$ | $\mathbb{R}^{N \times 256}$ |
| Predicted Temperature $\hat{T}$ | $\mathbb{R}^{N \times 1}$ |
| Predicted Stress $\hat{\sigma}$ | $\mathbb{R}^{N \times 6}$ |
| Predicted Displacement $\hat{u}$ | $\mathbb{R}^{N \times 3}$ |
| Predicted Heat Flux $\hat{q}$ | $\mathbb{R}^{N \times 3}$ |
