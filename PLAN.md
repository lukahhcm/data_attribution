# ICLR 实验计划：RHO/VF-1 与 Iterative VF-R

## 0. 论文要验证的核心论点

我们的理论观点是：多种经典 data attribution 方法，可以理解为在均匀样本权重

\[
\omega_0=\mathbf 1
\]

处，使用不同的 bilevel hypergradient estimator，估计同一个局部 selection quantity。
传统的 attribution-then-selection 只利用了真实 selection objective 在
\(\omega_0\) 附近的一阶局部代理。

其误差可以拆成两个来源：

1. **Hypergradient approximation error**：采用 IF、GU、VF 等不同方法近似
   bilevel hypergradient 时产生的误差。
2. **Non-local linearization error**：即使 \(\omega_0\) 处的方向估计准确，一次
   attribution 后直接选择较大的 subset，仍然可能离开该局部线性近似的有效区域。

本文实验集中验证第二点：在同一条 VF/RHO estimator 线上，相比只在均匀权重附近
进行一次全局选择，持续更新样本权重是否能够选出更好的固定训练子集。

因此，headline comparison 是：

\[
\boxed{\text{RHO/VF-1 vs. Iterative VF-R}}
\]

IF 和 GU 不进入第一阶段主实验。这样可以避免 inverse Hessian、截断迭代和
Adam 分母等额外数值问题干扰对 non-local linearization error 的验证。

---

## 1. 统一符号和方法命名

全文统一使用以下符号：

- \(n\)：candidate training set 中的数据总量；
- \(k\)：最终通过 global top-\(k\) 保留的数据量；
- \(q=k/n\)：最终 subset retention ratio；
- \(R\)：外层 sample-weight update rounds；
- \(r\in\{0,\ldots,R-1\}\)：外层更新轮次下标；
- \(T\)：两套 selector models 的总训练 epochs；
- \(S_r\)：第 \(r\) 个外层区间内两套模型的 SGD steps；
- \(B\)：Original RHO 的 candidate batch size；
- \(b\)：Original RHO 从每个 candidate batch 中选择的数据量；
- \(\omega_r\in\mathbb R^n\)：第 \(r\) 轮的全局样本权重；
- \(\hat\theta_r\)：weighted training model；
- \(\tilde\theta_r\)：包含 upper-validation objective 的辅助模型；
- \(\operatorname{TopK}_k(s)\)：根据全局 score vector \(s\) 选择 \(k\) 条数据。

方法命名固定为：

- `RHO/VF-1`：保持 \(\omega=\mathbf 1\)，只进行一次全局 attribution 和
  global top-\(k\)；
- `Iterative VF-R`：维护全局 \(\omega_r\)，进行 \(R\) 轮 weight update，最后
  global top-\(k\)；
- `Original RHO`：online training method，每个大小为 \(B\) 的 candidate batch
  内选择 top-\(b\)；
- `Uniform`：从整个 candidate set 中均匀选择固定的 \(k\) 条数据；
- `Uniform-online`：从每个 candidate batch 中随机选择 \(b\) 条用于当前更新。

不再使用 `VF-K` 表示迭代次数。`top-k` 中的 \(k\) 永远只表示最终选择的数据量，
\(R\) 永远只表示外层权重更新轮数。

---

## 2. 两套 selector models 的含义

### 2.1 Weighted training model：\(\hat\theta\)

\(\hat\theta\) 近似求解当前样本权重下的 weighted empirical-risk problem：

\[
\hat\theta(\omega)
\approx
\arg\min_\theta
\frac{1}{n}\sum_{i=1}^{n}\omega_i\ell_i(\theta).
\]

### 2.2 Auxiliary model：\(\tilde\theta\)

\(\tilde\theta\) 对应带 upper-validation objective 的辅助优化问题：

\[
\tilde\theta(\omega)
\approx
\arg\min_\theta
\left[
L_{\mathrm{val}}(\theta)
+
\alpha
\frac{1}{n}\sum_{i=1}^{n}\omega_i\ell_i(\theta)
\right].
\]

两套模型必须满足：

- 使用相同 architecture；
- 使用相同 initial parameters；
- 分别维护独立 optimizer 和 learning-rate scheduler；
- 训练中不共享参数或 optimizer state。

初始化形式为：

```python
theta_hat = make_model()
theta_tilde = deepcopy(theta_hat)
optimizer_hat = make_optimizer(theta_hat)
optimizer_tilde = make_optimizer(theta_tilde)
```

这样 reducible-loss difference

\[
s_i
=
\ell_i(\hat\theta)-\ell_i(\tilde\theta)
\]

主要反映当前 VF/RHO selection quantity，而不会混入两套模型的容量差异。

---

## 3. 核心比较的精确定义

### 3.1 RHO/VF-1：一次全局选择

RHO/VF-1 不是只训练一步模型。这里的 “1” 表示只进行一次外层 sample-selection
update。

具体过程是：

1. 初始化

   \[
   \omega_0=\mathbf 1.
   \]

2. 在固定的 \(\omega_0\) 下，正常训练 \(\hat\theta\) 和 \(\tilde\theta\)，总预算为
   \(T\) 个 epochs。
3. 训练结束后，对整个 candidate set 计算一次全局 score：

   \[
   s_i^{\mathrm{one}}
   =
   \ell_i(\hat\theta_T)-\ell_i(\tilde\theta_T).
   \]

4. 选择固定 subset：

   \[
   \mathcal S_k^{\mathrm{one}}
   =
   \operatorname{TopK}_k
   \left(s^{\mathrm{one}}\right).
   \]

因此，这个方法对应“在均匀权重处得到一个局部 attribution direction，然后直接走到
最终 top-\(k\) subset”的传统 attribution-then-selection 过程。

### 3.2 Iterative VF-R：多轮全局权重更新

Iterative VF-R 使用完全相同的总模型训练预算 \(T\)，但将该预算划分为 \(R\) 个
外层区间：

\[
\sum_{r=0}^{R-1}S_r=S_{\mathrm{total}}.
\]

每个 round 的过程是：

1. 使用当前 \(\omega_r\)，继续训练 \(\hat\theta_r\) 和 \(\tilde\theta_r\)；
2. 在整个 candidate set 上计算：

   \[
   s_i^{(r)}
   =
   \ell_i(\hat\theta_r)-\ell_i(\tilde\theta_r);
   \]

3. 对 score 做全局中心化和 RMS normalization：

   \[
   \bar s^{(r)}
   =
   \frac{s^{(r)}-\operatorname{mean}(s^{(r)})}
   {\sqrt{\operatorname{mean}[(s^{(r)}-\operatorname{mean}(s^{(r)}))^2]}+\epsilon};
   \]

4. 更新全局样本权重并投影：

   \[
   \omega_{r+1}
   =
   \Pi_{\Omega}
   \left[
   \omega_r+\eta_\omega\bar s^{(r)}
   \right],
   \]

   其中

   \[
   \Omega
   =
   \left\{
   \omega\in\mathbb R_+^n:
   \sum_{i=1}^{n}\omega_i=n
   \right\}.
   \]

完成 \(R\) 轮后：

\[
\boxed{
\mathcal S_k^{\mathrm{iter}}
=
\operatorname{TopK}_k(\omega_R)
}
\]

这里 \(\alpha_r\) 控制 \(\tilde\theta_r\) 的辅助优化问题，\(\eta_\omega\) 控制外层
样本权重更新。实现中不再额外用 \(\alpha_r\) 乘 normalized score，避免同时改变
auxiliary problem 和 outer step size。

### 3.3 Compute matching

两种固定-subset 方法必须保持以下条件一致：

- 相同模型初始化；
- 相同 optimizer 和 scheduler；
- 相同总 selector epochs \(T\)；
- 相同 global training batch size；
- 相同数据增强；
- 相同 \(\alpha\) schedule；
- 相同最终 \(k\)；
- 相同 evaluator protocol。

Iterative VF-R 会额外进行中间全局 scoring passes，因此严格的 wall-clock cost 不完全
相同。论文同时报告 selector GPU-hours；“compute matched”主要指两套模型的 SGD
training budget 相同，而不是隐藏额外 scoring cost。

---

## 4. Original RHO 与固定-subset 方法的区别

Original RHO 是 online selection-and-training：

\[
\mathcal S_{e,j}
=
\operatorname{TopK}_b
\left\{
s_i^{(e,j)}:i\in B_{e,j}
\right\},
\]

其中 \(B_{e,j}\) 是 epoch \(e\) 中第 \(j\) 个 candidate batch。

默认配置为：

\[
B=320,
\qquad
b=32.
\]

被选出的 \(b\) 条数据只用于当前 target-model update。下一个 candidate batch 会重新
评分、重新选择；Original RHO 不会先从整个数据集选出一个固定 subset，再使用该
subset 训练所有 epochs。

因此：

- `Original RHO` 不是 `RHO/VF-1`；
- `Original RHO` 也不是 `Iterative VF-R` 的 \(R=1\) 特例；
- Original RHO 报告 online training 的最终 test accuracy；
- RHO/VF-1 和 Iterative VF-R 报告固定 subset 上独立 evaluator 的 test accuracy。

Original RHO reproduction 默认让 scoring 和 target update 使用同一个 augmented
candidate view，并保留 train-mode scoring。由于 train-mode BatchNorm 会观察完整的
candidate batch，另外保留以下实现控制：

- `rho.selection_mode=eval`；
- `rho.score_view=deterministic`。

这些设置用于检查官方式实现细节是否影响结果，但不混入固定-subset headline
comparison。

---

## 5. 数据集、划分与噪声设置

第一阶段所有数据集均包含两个条件：

1. `clean`：noise rate \(\rho=0\)；
2. `noisy`：10% symmetric label noise，即 \(\rho=0.1\)。

噪声只施加在 candidate training split。Upper validation、selector development 和
test labels 始终保持 clean。每个被污染样本的新标签保证与原标签不同。

所有 split indices、clean labels、corrupted labels 和 corruption mask 都必须保存，
并在相同 dataset/seed 下由所有方法复用。

### 5.1 数据划分

| Dataset | Candidate train \(n\) | Upper validation | Selector dev | Final test |
|---|---:|---:|---:|---:|
| MNIST | 50,000 | 5,000 | 5,000 | official 10,000 |
| CIFAR-10 | 40,000 | 5,000 | 5,000 | official 10,000 |
| CIFAR-100 | 40,000 | 5,000 | 5,000 | official 10,000 |

采用 40k/5k/5k 而不是使用 test set 选择 reference checkpoint，是为了避免
Original RHO 类实现中潜在的 test leakage。

### 5.2 Retention ratios

| Dataset | 主实验 \(q=k/n\) | 对应 \(k\) |
|---|---|---|
| MNIST | \(\{0.1,0.2,0.5\}\) | \(\{5k,10k,25k\}\) |
| CIFAR-10 | \(\{0.1,0.2,0.5\}\) | \(\{4k,8k,20k\}\) |
| CIFAR-100 | \(\{0.2,0.5\}\) | \(\{8k,20k\}\) |

CIFAR-100 的 10% retention 作为可选 stress test，不列入第一轮必跑矩阵。

### 5.3 三个数据集在论文中的角色

#### MNIST：sanity check / appendix

- 快速验证更新方向、projection 和 index bookkeeping；
- 检查 noisy-label filtering 是否符合预期；
- 调试 \(\alpha\) 与 \(\eta_\omega\)；
- 不作为 ICLR 主结论的唯一证据。

#### CIFAR-10：主要开发与消融数据集

- 固定-subset 主结果；
- \(R\)、\(\alpha\)、\(\eta_\omega\) 和 selector budget 消融；
- weight、ranking 和 noise-removal trajectory；
- 第一轮所有实现决策在此确定。

#### CIFAR-100：困难场景验证

- 验证结论不只存在于简单数据集；
- 使用 CIFAR-10 上确定的最终配置；
- 不重复所有消融，避免无意义地扩大算力消耗。

---

## 6. 模型与训练配置

### 6.1 MNIST

Selector 和 evaluator 均使用两层 MLP：

```text
784 → 300 → 10
activation: Sigmoid
```

默认 selector 设置：

- optimizer：SGD；
- learning rate：0.01；
- global batch size：500；
- selector epochs：50；
- weight decay：0.01。

### 6.2 CIFAR-10/100

统一使用 CIFAR-style ResNet-18：

- 第一层为 \(3\times3\) convolution；
- stride 1；
- 不使用 ImageNet-style \(7\times7\) stem；
- 不使用初始 max-pooling；
- CIFAR-10 classifier 输出 10 类；
- CIFAR-100 classifier 输出 100 类。

默认 selector 设置：

| Setting | CIFAR-10 | CIFAR-100 |
|---|---:|---:|
| Selector epochs \(T\) | 100 | 120 |
| Global batch size | 128 | 128 |
| Optimizer | SGD | SGD |
| Learning rate | 0.1 | 0.1 |
| Momentum | 0.9 | 0.9 |
| Weight decay | \(5\times10^{-4}\) | \(5\times10^{-4}\) |
| Scheduler | 5-epoch warmup + cosine | 5-epoch warmup + cosine |

### 6.3 最终 evaluator

最终 evaluation model 必须：

- 从头随机初始化；
- 只使用保存的 selected indices；
- 不加载 \(\hat\theta_R\) 或 \(\tilde\theta_R\)；
- 所有 selection 方法使用相同 architecture；
- 使用相同 training epochs、augmentation、optimizer 和 scheduler；
- 使用 selector dev split 选择 checkpoint；
- official test 只用于最终一次报告。

主结果仍使用相同结构：

\[
\text{MLP on MNIST},
\qquad
\text{ResNet-18 on CIFAR}.
\]

可选的 architecture-transfer 实验为：

- ResNet-18 负责 selection；
- WideResNet-28-10 或 ResNet-34 从头训练并负责 evaluation。

该实验用于判断所选数据是否具有跨模型价值，属于加分项，不是第一轮必须完成。

---

## 7. \(\alpha\) 与 \(\omega\) 的更新配置

### 7.1 MNIST 的 \(\alpha\)

前 50% selector epochs 线性增加：

\[
\alpha:0.01\longrightarrow0.1,
\]

后 50% 保持 0.1。

### 7.2 CIFAR 的 \(\alpha\)

在初始模型上校准：

\[
c_\alpha
=
\frac{\|\nabla_\theta L_{\mathrm{val}}\|}
{\|\nabla_\theta L_{\mathrm{train}}\|+\epsilon}.
\]

前 50% selector epochs 线性增加：

\[
\alpha:0.1c_\alpha\longrightarrow3c_\alpha,
\]

后 50% 保持 \(3c_\alpha\)。DDP 实现中先对各卡 gradient 做平均，再计算 global
gradient norm，保证 \(c_\alpha\) 不随 GPU 数量发生系统性变化。

### 7.3 外层权重更新

主配置为：

\[
R=10,
\qquad
\eta_\omega=0.1,
\qquad
\omega_0=\mathbf1.
\]

每轮更新后投影到 non-negative simplex，保持：

\[
\omega_i\ge0,
\qquad
\sum_i\omega_i=n.
\]

---

## 8. 主实验矩阵

### 8.1 Fixed-subset headline table

每个数据集、noise condition 和 retention ratio 比较：

1. `Full-data`；
2. `Uniform` global top-\(k\)；
3. `RHO/VF-1`；
4. `Iterative VF-R`，主配置 \(R=10\)。

核心论文比较是：

\[
\operatorname{Acc}
\left(\mathcal S_k^{\mathrm{iter}}\right)
\quad\text{vs.}\quad
\operatorname{Acc}
\left(\mathcal S_k^{\mathrm{one}}\right).
\]

所有 headline results 使用 5 seeds。

### 8.2 Online RHO table

单独报告：

1. `Uniform-online`：每个 \(B=320\) candidate batch 随机选择 \(b=32\)；
2. `Original RHO`：每个 \(B=320\) candidate batch 按 reducible loss 选择 top-32。

Online table 是相关 baseline，不与固定-subset evaluator 的训练过程混为一谈。

### 8.3 首轮最小论文配置

| Dataset | Conditions | Retention | Seeds | 主要用途 |
|---|---|---|---:|---|
| MNIST | clean、10% noise | 10/20/50% | 3–5 | sanity check |
| CIFAR-10 | clean、10% noise | 10/20/50% | 5 | 主结果、完整分析 |
| CIFAR-100 | clean、10% noise | 20/50% | 5 | 困难场景验证 |

第一轮 selector 主矩阵共有：

\[
(3+3+2)\times2\text{ noise conditions}
\times2\text{ methods}\times5\text{ seeds}
=160
\]

个 `RHO/VF-1` / `Iterative VF-R` selector jobs，随后分别训练 evaluator。

---

## 9. 必须报告的结果与诊断量

### 9.1 最终性能

- test accuracy；
- upper-validation loss；
- selected-noise fraction；
- selected-clean percentage；
- selector GPU-hours；
- evaluator GPU-hours；
- 5 seeds 的 mean ± standard deviation；
- Iterative VF-R 相对 RHO/VF-1 的 paired difference；
- paired bootstrap confidence interval。

### 9.2 每轮 trajectory

在每个 weight-update round 保存：

- score vector \(s^{(r)}\)；
- sample weights \(\omega_r\)；
- 当前 top-\(k\) indices；
- 与第一轮 top-\(k\) 的 Jaccard overlap；
- 与第一轮 score 的 Spearman rank correlation；
- weight effective sample size：

  \[
  \operatorname{ESS}(\omega)
  =
  \frac{(\sum_i\omega_i)^2}{\sum_i\omega_i^2};
  \]

- selected-noise fraction；
- observed-label class counts；
- clean-label class counts；
- validation loss。

这些 trajectory 用于回答：

1. 多轮更新是否真的改变了 ranking；
2. 改变发生在前几轮还是持续发生；
3. ranking change 是否对应更少的 corrupted examples；
4. 性能提升是否只是权重过度集中造成的；
5. 在 clean data 上是否出现类别塌缩。

---

## 10. 消融实验

除特别说明外，消融固定在：

\[
\text{CIFAR-10},
\quad
10\%\text{ label noise},
\quad
q=0.2,
\quad
3\text{ seeds}.
\]

### 10.1 外层更新轮数

保持总 selector epochs \(T=100\) 不变：

\[
R\in\{1,2,5,10,20\}.
\]

这是最重要的消融，用于检查收益是否随多轮 refinement 出现，以及何时饱和。

### 10.2 \(\alpha\) schedule

- constant；
- linear continuation；
- geometric continuation。

### 10.3 \(\alpha_{\max}\)

\[
\alpha_{\max}
\in
\{1,3,10\}c_\alpha.
\]

### 10.4 Outer weight step size

\[
\eta_\omega
\in
\{0.03,0.1,0.3\}.
\]

### 10.5 Selector training budget

\[
T\in\{50,100,200\}\text{ epochs}.
\]

该消融用于排除 Iterative VF-R 的优势只是中间模型训练不足或过度训练造成的。

### 10.6 第二阶段实验

首轮结果稳定后再考虑：

- 20% 或更高 label noise；
- feature corruption；
- long-tail class imbalance；
- ResNet-18 selection 到 WideResNet-28-10 evaluation 的跨模型迁移。

这些实验不应阻塞 clean + 10% noisy 的第一轮主结果。

---

## 11. 结果如何支持理论论点

仅仅观察到 Iterative VF-R 最终准确率更高，不能单独“证明”VF score 等于
\(\omega=\mathbf1\) 处的一步近似；该对应关系由理论推导建立。

实验验证的是这项推导的算法后果：

1. `RHO/VF-1` 使用均匀权重附近的一次局部 score；
2. `Iterative VF-R` 允许模型状态、score 和 \(\omega_r\) 共同演化；
3. 若多轮方法稳定取得更优 subset，并且 ranking trajectory 显示其逐渐偏离第一轮
   ranking，那么结果支持 non-local linearization error 在实际 selection 中确实重要；
4. 若 \(R=1\) 与 RHO/VF-1 接近，而 \(R>1\) 后逐渐改善，这将是最直接的实验模式；
5. 若提升同时对应更低 selected-noise fraction，说明 improvement 具有可解释的
   data-cleaning mechanism，而不仅是 evaluator 方差。

主要论证链条写成：

\[
\text{one-step local attribution}
\longrightarrow
\text{ranking changes across rounds}
\longrightarrow
\text{better fixed subset}
\longrightarrow
\text{better from-scratch evaluation}.
\]

---

## 12. 多 GPU 与集群执行方案

### 12.1 单个 selector run 的 DDP

- \(\hat\theta\) 和 \(\tilde\theta\) 都使用 DistributedDataParallel；
- global batch size 在各 rank 间均分；
- \(\omega\in\mathbb R^n\) 在每个 rank 上复制；
- 每轮 scoring 时，各 rank 按 global candidate index 累加 score sum 和 count；
- 使用 all-reduce 得到完整全局 score；
- 每个 rank 执行相同 weight update 和 projection。

### 12.2 大规模 sweep 的首选方式

对于 ResNet-18 规模，优先采用：

\[
\boxed{\text{one independent seed/configuration per GPU}}
\]

而不是强制每个小模型占用很多 GPU。这样更容易把集群利用率拉满，也减少 DDP
communication overhead。

Original RHO 每个 run 保持单 GPU，以维持精确的 \(320\to32\) online batch
semantics；不同 seeds/configurations 在不同 GPU 上并行。

### 12.3 Checkpoint 与恢复

每个 outer round 后保存：

- 两套 selector model states；
- 两个 optimizer states；
- 两个 scheduler states；
- AMP scaler states；
- 当前 \(\omega_r\)；
- 当前 score；
- epoch 和 round counters。

集群中断后从最近一个 round boundary 恢复。

---

## 13. 分阶段执行顺序

### Phase A：代码正确性与 smoke test

1. MNIST clean，缩短 epochs，检查训练能否结束；
2. MNIST 10% noise，检查保存的 corruption mask；
3. 检查 \(\sum_i\omega_i=n\) 和 \(\omega_i\ge0\)；
4. 检查 global top-\(k\) 数量和 indices 唯一性；
5. 用 CIFAR-10 单 GPU 和 2-GPU DDP 比较 score/order 是否基本一致；
6. 检查中断恢复是否能继续到下一 round。

### Phase B：小规模开发

1. CIFAR-10，10% noise，\(q=0.2\)；
2. 运行 3 seeds；
3. 比较 RHO/VF-1、\(R=2,5,10\)；
4. 确定 \(\alpha\)、\(\eta_\omega\) 和数值稳定性；
5. 冻结最终配置。

### Phase C：完整主实验

1. CIFAR-10 clean + noisy 全 retention ratios、5 seeds；
2. CIFAR-100 clean + noisy、5 seeds；
3. MNIST clean + noisy sanity results；
4. Uniform、Full-data 和 online RHO baselines；
5. 所有 fixed-subset 方法训练独立 evaluator。

### Phase D：消融与附加实验

1. CIFAR-10 上完成 \(R\) 消融；
2. 完成 \(\alpha\)、\(\eta_\omega\) 和 selector budget 消融；
3. 根据主结果决定是否增加 higher noise、feature corruption、class imbalance；
4. 资源允许时进行 architecture transfer。

---

## 14. 算力不足时的优先级

按论文价值排序：

1. CIFAR-10 + ResNet-18 主结果与 \(R\) 消融；
2. CIFAR-100 + ResNet-18 困难场景；
3. CIFAR-10 的 trajectory figures；
4. MNIST sanity check；
5. Original RHO online table；
6. 跨模型迁移；
7. 更多 online RHO 实现变体。

最简洁的论文叙事是：

\[
\text{MNIST：算法与实现可运行},
\]

\[
\text{CIFAR-10：完整主结果、机制分析和消融},
\]

\[
\text{CIFAR-100：困难场景验证}.
\]

宁愿优先保证 CIFAR-10/100 的 seeds、matched budgets 和 evaluator protocol，也不要
为了覆盖更多 online/global 组合而削弱核心比较。
