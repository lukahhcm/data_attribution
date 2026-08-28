# 实验方案更新：多轮双层选择与 RHO Batch-Size 分解

更新日期：2026-08-27。本文件记录本轮论文实验协议更新；根目录 `PLAN.md` 继续作为完整技术规范，不在此覆盖。

## 0. 论文理论范围与必须保留的 one-shot baselines

论文的第一层 insight 不是只讨论 VF--RHO，而是把三类流行 attribution 方法与求解同一个 bilevel data-weighting outer gradient 的三类近似建立一一对应：

| attribution family | bilevel approximation | one-shot method |
|---|---|---|
| Influence Function | implicit differentiation | `IF-1` |
| TSLOO / SGD-Influence | gradient unrolling | `TSLOO-1` |
| ideal RHO-LOSS | value-function penalty / F2SA | `VF-Persistent-1` / `RHO-Persistent-1` |

理论映射在相同的 `f`、`g`、数据和 reference `omega` 上比较；canonical attribution point 是全 1，固定预算 selection comparison 另行报告 `omega_unif=(B/n)1`，两者不得混写。正式多轮实验只沿 VF--RHO 支线展开，但 `IF-1` 与 `TSLOO-1` 必须保留在论文和 one-shot 实验中。

主矩阵前增加 diagnostic pre-wave：MNIST、CIFAR-10、10% noise、seed 1，对比 `IF-1 / TSLOO-1 / VF-Persistent-1 / RHO-Persistent-1`。`IF-1` 使用 CG iHVP；`TSLOO-1` 在 MNIST 使用完整 recurrence，在 CIFAR 使用预注册 checkpoint approximation。所有方法选一次 Top-B 后从共同 target warm-start 训练到相同 `H`，同时报告 attribution auxiliary compute。

## 1. 研究问题

1. **RQ1（uniform utility vs. optimized utility）**：一次从均匀权重点出发的 attribution 更新，与沿训练轨迹持续更新 continuous `omega` 相比，哪一个能得到更好的 Top-10% 数据？
2. **RQ2（bi-level steps scaling）**：完整的 inner update、outer score、continuous weight update 和 Top-B deployment 重复 1--5 轮时，最终模型是否随轮数改善？
3. **RQ3（technical choices）**：practical RHO 的 batch-local quota、立即训练、online 重打分和 frozen holdout model 各自贡献多少？

关键约束：`omega` 是固定预算集合上的连续外层状态，`score` 是更新方向；只有真实训练目标模型时才计算 `Top-B(omega)`。下一轮更新不得把二值 Top-B mask 当作梯度变量。

## 2. 固定预算与数据设置

```text
W_B = {omega in [0,1]^n : sum_i omega_i = B}
B   = 0.1 n
S_r = Top-B(omega_r)
```

所有成对方法固定数据划分、噪声 realization、初始化、warm-start checkpoint、目标模型、AdamW、scheduler、target epoch 数和每 epoch 训练样本数。只用 development set 做配置选择；test set 只在预注册终点评估。

| 数据集 | candidate / development / test | warm start | block `L` | horizon `H=5L` |
|---|---:|---:|---:|---:|
| MNIST | 50,000 / 5,000 / 5,000 | 10 | 30 | 150 |
| CIFAR-10 | 40,000 / 5,000 / 5,000 | 20 | 80 | 400 |
| CIFAR-100 | 25,000 / 25,000 / 10,000 | 20 | 100 | 500 |

主设置为 10% symmetric label noise、seeds 1--3；clean 是必要对照。20% 和 30% noise 后续扩展。

## 3. RQ1/RQ2：固定总训练量的五段 block protocol

共享 warm start 后，target 训练为五个等长 block。比较 `R in {1,2,3,4,5}`：前 `R` 个 boundary 更新 selector，第 `R` 次后冻结 selector 与 Top-B，但 target 始终继续到相同的 `H=5L`。每个 boundary 保存完整 checkpoint，并建立 freeze-`R` 分支。

| 方法 | continuous memory | boundary 操作 |
|---|---:|---|
| Uniform-Block-R | 否 | 前 R 次随机重选 Top-10%，随后冻结 |
| RHO-Reset-R | 否 | 每次从 `omega_unif` 根据 practical RHO score 更新 |
| RHO-Persistent-R | 是 | 从上一轮 `omega_r` 继续更新 |
| VF-Reset-R | 否 | 每次从 `omega_unif` 做标准 VF/F2SA update |
| VF-Persistent-R | 是 | 更新并投影同一个 continuous `omega_r` |

`R=1` 定义 one-shot attribution-then-selection。每个 VF round 必须包括固定 inner passes、全局 VF score、一次 continuous outer update 与投影、Top-B deployment。Strict VF 维护：

```text
theta_hat(omega)   <- argmin g(omega, theta)
theta_tilde(omega) <- argmin f(theta) + alpha_r g(omega, theta)
score_i            = ell(theta_hat; z_i) - ell(theta_tilde; z_i)
alpha_r            = alpha_0 (1+r)^p
```

主实验使用固定 inner passes；自适应收敛仅用作 RQ1 reference evaluator。每个 boundary 保存 target、两个 inner models、optimizer/scheduler、continuous omega、score、Top-B indices、BN buffers、RNG 和 dev metrics。

## 4. RQ3：RHO 公式与在线协议拆解

依次加入：

1. `RHO-Global-TopB`：block 开始全局打分，取全局 Top-10%，整块训练；
2. `RHO-GlobalScore-LocalQuota`：复用 block-start score，但每个 candidate batch 取局部 Top-10%；
3. `RHO-Stale-Online`：局部选择后立即更新 target，但本 block score 固定；
4. `RHO-Batch-Faithful`：每个 candidate batch 用最新 target 重打分、局部 Top-10% 后立即训练。

在相同协议点比较三种 score：strict VF；current target + updating joint comparator；current target + frozen holdout IL。由此分别识别 score approximation 与 batch-online protocol 的收益。

## 5. 优先实验：RHO batch-size scaling

固定 local retention=10%，`n_b` 表示被选择并立即用于一次 target update 的 batch。一般令 `n_B=10n_b`，按 2 倍递增扩展在线 batch scale；每个数据集还显式加入 `n_B=n, n_b=0.1n` 的全数据集端点：

| selected `n_b` | candidate `n_B` | local retention |
|---:|---:|---:|
| 32 | 320 | 10% |
| 64 | 640 | 10% |
| 128 | 1,280 | 10% |
| 256 | 2,560 | 10% |
| 512 | 5,120 | 10% |
| 1,024 | 10,240 | 10% |
| 2,048 | 20,480 | 10% |

最终使用的数据集特定轴为：

| 数据集 | 几何递增的 `n_b` | 全数据集端点 `(n_B,n_b)` |
|---|---|---|
| MNIST | 32 / 64 / 128 / 256 / 512 / 1,024 / 2,048 / 4,096 | (50,000, 5,000) |
| CIFAR-10 | 32 / 64 / 128 / 256 / 512 / 1,024 / 2,048 | (40,000, 4,000) |
| CIFAR-100 | 32 / 64 / 128 / 256 / 512 / 1,024 / 2,048 | (25,000, 2,500) |

最后一个不完整 candidate batch 按 10% 四舍五入，再执行 epoch exact-budget correction，保证总计恰为 `0.1n` 且无重复。增大 `n_b` 会保持 target examples 不变但减少 optimizer steps；这是 batch-size scaling 干预的一部分，必须同时报告 examples、steps 和 wall-clock，不能把曲线解释为只改变 oversampling diversity。

每一点同时运行：

- `RHO-Batch-Faithful(n_B,n_b)`；
- `Uniform-Batch-Matched(n_B,n_b)`，共享 candidate order、local budget、立即更新和 target steps，只替换排序。

reference/IL model 的训练 batch 固定为 32；同一 dataset/seed/split 的完整在线 batch-size 轴读取同一个 reference checkpoint。

执行 waves：

1. Wave 1：MNIST、CIFAR-10，10% noise，seed 1，完整 batch-size 轴，RHO 与 matched Uniform；
2. Wave 2：CIFAR-100，10% noise，seed 1，完整 batch-size 轴；
3. Wave 3：上述曲线扩展到 seeds 1--3，并补 clean setting；任何后续代表性点选择只用 development；
4. Wave 4：启动 block protocol 的 `R=1..5` 主矩阵。

指标：final test accuracy、best development loss、selected-noise rate、class entropy、cumulative unique coverage、adjacent turnover、target examples/steps、scoring examples、wall time、peak GPU memory。

## 6. 结论边界

- Persistent-R 随 R 改善且优于 Reset-R：支持沿轨迹维护 optimized continuous weights。
- Reset-R 改善而 Persistent-R 不改善：收益来自 fresh score，而不是 outer memory。
- faithful RHO 仅在较小 batch 领先 global RHO：优先归因于 local quota、online adaptation 或 step granularity，不能直接归因于更准确的 VF 近似。
- 计算量不同时，同时报告 accuracy-vs-target-data 与 accuracy-vs-total-compute。
