# Data Attribution V2：论文论断与实验执行计划

更新日期：2026-08-26。本文件取代 `plan_0820.md`，是下一阶段唯一的实验规划依据。

## 1. 中心论断与证据链

论文的中心问题不是“VF 是否击败所有 baseline”，而是：

> data attribution 给出一个固定状态下的局部方向；data selection 需要沿真实训练轨迹，
> 在固定预算约束内多次更新这个方向。多轮相对单轮的收益是否存在、来自哪个状态组件，
> 以及 practical RHO 的近似为何有效？

证据必须按顺序成立：

1. **局部正确性**：在相同目标、权重和 inner solutions 下，ideal RHO loss difference
   与 finite-penalty VF/F2SA 下降方向一致。
2. **单轮基准**：一次从 budget-feasible uniform point 出发的 VF/RHO 更新，定义为
   one-shot attribution-then-selection。
3. **多轮收益**：persistent VF-M 在相同 target 数据量和 optimizer steps 下优于 VF-1。
4. **机制拆解**：收益可被分解为 target adaptation、inner tracking、outer memory 和
   repeated Top-B rounding。
5. **近似前沿**：practical RHO 的 frozen IL、nonconverged target 和 batch-local scoring
   如何改变 fidelity、accuracy 与 compute。

如果第1步不成立，后续 VF 只能作为 heuristic 报告；如果第3步不成立，论文不能声称
multi-round optimization 有收益，但仍可报告 RHO/VF 对应关系和近似前沿。

## 2. 统一问题与状态

候选数据为 `D_tr={z_i}_{i=1}^n`，clean holdout 为 `D_val`：

```text
g(omega, theta) = (1/n) sum_i omega_i ell(theta; z_i)
f(theta)        = (1/|D_val|) sum_{z in D_val} ell(theta; z)
```

固定预算集合：

```text
W_B = {omega in [0,1]^n : sum_i omega_i = B}
```

必须区分：

- `omega_r`：persistent continuous outer state；
- `score_r`：当前状态下的 negative outer-gradient estimate；
- `S_r=Top-B(omega_r)`：下一个真实 target block 使用的离散子集；
- `theta_target`：真实训练并最终评估的模型；
- `theta_hat`、`theta_tilde`：只服务于 VF estimator 的两条 inner branches。

预算实验的 uniform/reset point 固定为：

```text
omega_unif = (B/n) * 1
```

全1向量只保留为“所有数据完整存在”的 attribution reference；它不用于10%预算下的
reset/persistent 比较。

## 3. 新版 VF/F2SA 规范

### 3.1 两个 inner problems

在 outer round `r`：

```text
theta_hat(omega_r)          = argmin_theta g(omega_r, theta)
theta_tilde(omega_r)        = argmin_theta f(theta) + alpha_r g(omega_r, theta)
```

VF score：

```text
score_i = ell(theta_hat; z_i) - ell(theta_tilde; z_i)
```

由于

```text
[grad_omega L^{alpha,*}(omega)]_i
  = (alpha_r/n) [ell(theta_tilde;z_i)-ell(theta_hat;z_i)]
```

代码用 `omega <- omega + eta * standardized(score)`，再投影回 `W_B`。

### 3.2 沿用原稿的 alpha；外部符号只作内部校验

论文与本 plan 始终使用原稿的 `alpha`：

```text
alpha_r = alpha_0 * (1+r)^p
```

实现配置使用语义中性的 `f2sa.penalty`，其数值就是文稿中的 `alpha_r`。不要把
外部论文的变量名写进正文。为防止公式或代码错位，内部维护如下映射：

| 本文 | F2SA 原文 | 在本文中的角色 |
|---|---|---|
| `omega_r` | outer variable `x_k` | continuous example weights |
| `theta_tilde` | penalized-inner tracker `y_k` | minimize `f + alpha_r g` |
| `theta_hat` | lower-level tracker `z_k` | minimize `g` |
| `alpha_r` | penalty sequence `lambda_k` | value-function penalty multiplier |

RHO-LOSS 的概念映射同样只留在 plan：ideal RHO 的 training-only model 对应
`theta_hat`，joint holdout-plus-weighted-training model 对应 `theta_tilde`；practical RHO
则用当前 `theta_target` 和冻结的 `theta_IL` 替代这两条收敛分支。

preflight 比较：

- `alpha_0 in {1, 10}`；
- `p in {0, 0.25}`；
- `p=0` 明确标为 fixed finite-penalty approximation；
- 主实验只保留一个 validation-selected schedule。

不能因为 score standardization 消除了全局尺度，就忽略 alpha 对 `theta_tilde` 解的影响。

### 3.3 inner tracking 与停止

两条 inner branches 在第一次 round 从相同模型初始化和相同 BN buffers 出发，此后分别
persistent warm start。inner optimization：

- BN running mean/variance 固定；BN affine parameters 正常学习；
- `theta_hat` 最小化 fixed-denominator weighted candidate risk；
- `theta_tilde` 最小化 clean holdout risk + `alpha_r` 倍 weighted candidate risk；
- candidate 与 holdout 的样本数、optimizer steps 分别记录。

preflight 比较 inner passes `{1,5,10,20,50}`。停止规则只用 validation/inner 信息：

- 两个 objective 的相对改善均小于 `1e-3`，连续3次检查；
- 相邻全局 score cosine `>0.99`；
- 相邻 Top-B Jaccard `>0.95`；
- 最多50个 candidate passes。

主实验使用 preflight 选出的最小稳定 passes 或停止规则。不得用 test accuracy 决定。

### 3.4 omega 参数化

主方案：direct continuous omega + capped-simplex projection。

消融：unconstrained logits `a`，通过

```text
omega_i = sigmoid(a_i + c(a)),  sum_i omega_i = B
```

求一维 shift `c(a)` 保证预算。logit 更新必须使用 budget-calibrated sigmoid 的链式梯度；
direct 与 logit learning rate 分开调节。

## 4. 单轮与多轮的精确定义

### 4.1 共享 warm start

所有方法共享：

- target initialization；
- 随机预算子集 `S0`；
- 在 `S0` 上的前 `E0` 个 target epochs；
- optimizer、scheduler、数据顺序和 checkpoint rule。

`E0` 只通过 uniform-fixed 的 development curve 选择：在候选 `{5,10,20}` 中取首次达到
预注册 plateau criterion 的值；每个数据集选定后冻结。保存 `S0`，所有 paired methods
直接读取同一文件。

### 4.2 One-shot

warm start 后：

1. 在 `omega_unif` 计算一次 score；
2. 更新一次 omega 并取 `S1=Top-B(omega_1)`；
3. 剩余 target epochs 始终训练 `S1`；
4. 不再更新 score、inner models 或 omega。

得到 `RHO-1` 和 `VF-1`。

### 4.3 Multi-round

warm start 后，每 `tau` 个真实 target epochs：

1. 将 inner models 跟踪到当前 `omega_r` 的稳定解；
2. 重新计算 score；
3. 更新 persistent `omega_r -> omega_{r+1}`；
4. 取 `S_{r+1}=Top-B(omega_{r+1})`；
5. 用该子集真实训练下一个 target block。

候选 update interval：

```text
tau in {1, 5, 10, 20, infinity}
```

`tau=infinity` 即 one-shot。先在 MNIST 10% noise、seed 1 筛选，再把最有代表性的两个
finite intervals 带入完整 paired-seed 实验。interval 改变 selector 更新次数，因此同时报告
accuracy-vs-data 和 accuracy-vs-compute。

## 5. RHO 协议

### 5.1 Faithful practical RHO

- IL model 只在 clean holdout 上训练；按 development/holdout loss 选择 checkpoint；
- IL model 在 target training 前冻结，irreducible losses 可缓存；
- 每步随机读取 large batch `n_B=320`；
- 用当前 target loss - frozen IL loss 打分；
- 选择 top `n_b=32`，立即更新 target；
- selection 使用 large-batch BN statistics，target update 使用 selected-batch BN statistics；
- 两次 scoring 都不提交 selection 阶段产生的 BN running-buffer 改变。

这个方法命名为 `rho_batch_faithful`。

### 5.2 RHO extensions

- `rho_global_one_shot`：全 candidate 打分一次；
- `rho_global_multi`：按 tau 全局重新打分；
- `rho_joint_comparator`：更新 joint holdout+candidate comparator，即 ideal-RHO 近似；
- `rho_frozen_il`：practical frozen comparator。

global RHO 是本文扩展，不标为原论文算法。

### 5.3 RHO batch-size 支线

固定 retention=10%，比较 matched pairs：

```text
(n_B, n_b) in {(160,16), (320,32), (640,64), (1280,128)}
```

这样只改变 batch scale，不改变选择比例。先在 MNIST/CIFAR-10 10% noise、seed 1 筛选，
再对前两名做3 seeds。预算比例 sweep 是另一实验，不能与 batch-size sweep 混合解释。

## 6. 三个 RQ 与实验

### RQ1：loss difference 是否是可用的 VF/F2SA 局部方向？

**RQ1a：数学正确性。** 在2--10维 strongly-convex toy problem 比较：

- analytic hypergradient；
- finite difference；
- full autograd unroll；
- converged finite-penalty VF loss difference。

必须满足符号一致、cosine `>0.99`、relative error 在数值容差内。

**RQ1b：神经网络中的近似误差。** 在相同 omega/checkpoint 比较：

1. converged VF；
2. truncated VF；
3. ideal RHO / updating joint comparator；
4. practical RHO / frozen IL；
5. current nonconverged target 替代 theta_hat。

主要指标：signed cosine、Spearman、Top-B Jaccard、score norm、inner residual、
selected-noise fraction。final accuracy 只作次要后果。

### RQ2：多轮是否优于单轮？

核心方法矩阵：

| ID | 方法 | 回答的问题 |
|---|---|---|
| U-fixed | warm start 后固定随机子集 | 不重新选择 |
| U-round | 每 block 重采随机子集 | 单纯 reselection |
| RHO-1 | practical/global RHO 一次 | practical one-shot |
| VF-1 | verified VF 一次 | 理论 one-shot |
| RHO-M | practical/global RHO 多轮 | cheap multi-round |
| VF-M | persistent verified VF | full multi-round |

主 estimand：

```text
Delta_multi = Accuracy(VF-M) - Accuracy(VF-1)
```

次要 estimands：RHO-M - RHO-1、VF-1 - U-fixed、VF-M - U-round。

比较必须 paired by dataset split、noise realization、target initialization、S0 和 seed。
只有 paired CI 为正且 accuracy-vs-compute 不被单轮支配时，才声称 multi-round benefit。

### RQ3：多轮收益与 practical RHO 近似分别来自什么？

从 verified VF-M 出发一次只改变一个组件：

1. `short-inner`：减少 inner passes；
2. `frozen-inner`：inner models 不再跟踪；
3. `current-target-as-hat`：用 target 替代 theta_hat；
4. `frozen-joint`：joint comparator 停止更新；
5. `holdout-only-IL`：替换为 practical IL；
6. `batch-local`：global scoring 改 large-batch scoring；
7. `reset-omega`：每轮回到 omega_unif；
8. `delayed-rounding`：更新 omega 但中途不改变 target subset；
9. `sigmoid-omega`：projection 改 budget-calibrated logits。

四个机制对照直接对应多轮状态：

- frozen target -> target adaptation；
- frozen inner -> inner tracking；
- reset omega -> outer memory；
- delayed Top-B -> repeated rounding。

报告每一级相对上一级的 paired accuracy delta、score agreement、额外 FLOPs/wall-clock 和显存。

## 7. 数据集、噪声与预算的阶段顺序

固定 split（official test 始终独立）：

| Dataset | Candidate | Clean holdout | Development | Test |
|---|---:|---:|---:|---:|
| MNIST | 50,000 | 5,000 | 5,000 | 10,000 |
| CIFAR-10 | 40,000 | 5,000 | 5,000 | 10,000 |
| CIFAR-100 | 25,000 | 20,000 | 5,000 | 10,000 |

CIFAR-100 保留旧实验的25k candidate scale，但不再把同一25k validation 同时当
development；20k 只进入 upper/IL objective，5k 只用于 checkpoint、early stop 和调参。

不得一次生成旧式全矩阵。按以下 gate 逐步扩展：

### Stage A：Correctness

- toy exact-gradient tests；
- MNIST、10% noise、seed 1；
- alpha、inner passes、omega parameterization preflight。

### Stage B：RQ1 approximation ladder

- MNIST、CIFAR-10；
- 10% noise；
- seeds 1/2/3；
- 不做 retention sweep。

### Stage C：RQ2 interval screening

- MNIST、10% noise、seed 1；
- tau `{1,5,10,20,infinity}`；
- 选择两个 finite intervals，不根据 test set 选择。

### Stage D：RQ2 main comparison

- datasets：MNIST、CIFAR-10、CIFAR-100；
- noise：0%、10%；
- seeds：1/2/3；
- methods：6个核心方法，multi 方法只用筛选出的2个 intervals。

### Stage E：RQ3 nested ablations

- MNIST、CIFAR-10；
- 10% noise；
- 先 seed 1，再对改变结论的组件做3 seeds。

### Stage F：robustness

只有前述结论稳定后：

- noise 扩展到20%、30%；
- retention 扩展到5%、15%、20%；
- budget sweep 仅保留 U-round、VF-1、VF-M、best practical RHO；
- RHO batch-size 支线独立报告。

## 8. 指标与统计

主要结果：

- final/best-development-selected test accuracy；
- accuracy versus real target examples；
- accuracy versus target optimizer steps；
- accuracy versus selector compute/wall-clock。

机制诊断：

- selected-noise fraction；
- class coverage、entropy、min/max count；
- continuous omega min/max/sum/ESS；
- adjacent score cosine；
- Top-B Jaccard、turnover、cumulative coverage；
- two-inner objectives、gradient/stationarity proxy、passes；
- alpha、outer step size、projection residue；
- target/inner/scoring examples、steps、wall-clock、peak GPU memory。

统计：

- 所有核心比较按 seed 配对；
- 报告 mean、sample SD、paired delta 和 bootstrap/t interval；
- 不用“2/3 seeds 赢”替代效应量；
- test set 每个 run 只在最终 checkpoint protocol 确定后评估。

## 9. 旧实验的复用规则

### 可复用基础设施

- deterministic dataset split 和 symmetric noise realization；
- `data_indices_and_noise.npz`；
- MNIST MLP、CIFAR ResNet-18；
- target optimizer/evaluation、logging；
- capped-simplex projection；
- uniform baseline 的历史结果范围。

### 可复用但必须重新配对运行

- `uniform_epoch`、`uniform_batch`：用来做新 runner regression/sanity check；不能与新方法
  直接做 paired significance，因为旧 run 没有共享新版 S0 warm start。
- holdout-only IL model 的训练代码：协议可复用；checkpoint 只有 config/hash 完全一致时复用。

### 仅作历史诊断

- `rho_batch_matched`：旧配置没有 faithful selection-batch BN，必须重跑；
- `rho_epoch_persistent_u1` 与 `vf_epoch_persistent_u1`：保留掉点作为 motivation；
- fairness audit：只证明旧矩阵预算/metadata 自洽，不证明 estimator 正确。

### 不进入新论文结论

- online-K；
- persistence/block-K；
-旧 strict-VF U1/UB；
- fresh evaluator 旧结果。

fresh evaluator 未来只可作为次要问题：最终 selection solution 是否能迁移到新初始化/模型；
不能用来证明 multi-round optimization 本身有效。

## 10. Go / No-Go

进入完整 GPU 主实验需同时满足：

1. toy exact-gradient tests 通过；
2. omega 全程满足预算约束；
3. ideal RHO 与 converged VF 的方向一致；
4. inner stopping 后 score 对增加 passes 不敏感；
5. shared S0、target init 和 data order 可审计；
6. test set 未参与任何选择；
7. 单次 run 可从 config、seed、S0 和 checkpoints 完全复现。

如果 VF 仍低于 RHO，先报告 gradient agreement、inner residual、finite-alpha bias、score stability、
projection/rounding 和 compute；全部正常后，才可把差距解释为 practical approximation 的收益。

## 11. 服务器实现 TODO

当前 `data_attribution_v2` 已包含数学与协议核心、配置、correctness tests 和旧结果摘录，
但尚未包含可直接运行完整神经网络实验的端到端 runner。以下工作在服务器上按顺序完成；
前一级未通过时，不启动后一级实验。

### 11.1 建立并验证运行环境

- 使用 Python 3.11 或更高版本建立独立环境；
- 安装 `data_attribution_v2` 的 dev dependencies；
- 记录 Python、PyTorch、CUDA、cuDNN、GPU、driver 和 git/code snapshot；
- 将数据根目录、artifact 根目录和设备设置改为服务器路径，但不修改实验语义；
- 运行：

```bash
cd data_attribution_v2
python3.11 -m pip install -e '.[dev]'
pytest -q
python3.11 scripts/run_toy_gate.py
```

完成标准：全部单测通过，toy gate 的三组 cosine 均大于 `0.999`，finite-difference
relative error 小于 `1e-6`。

### 11.2 实现统一端到端 runner

新增一个统一入口，例如 `scripts/run_experiment.py`，不得为每个方法复制一套训练代码。
runner 必须组合现有模块，而不是重新定义其语义：

- `data.py`：稳定 index、candidate/holdout/development/test split 和 label noise；
- `models.py`、`training.py`：模型、target optimizer、scheduler 和 evaluation；
- `protocol.py`：shared warm start、one-shot/multi-round update boundaries；
- `omega.py`：persistent continuous state、projection/sigmoid 和 Top-B view；
- `selector.py`、`f2sa.py`：两条 inner branches、停止规则和 VF score；
- `rho.py`：faithful practical RHO batch scoring；
- `io.py`：config、checkpoint、metrics 和 provenance。

runner 需实现六个核心方法：

```text
uniform_fixed, uniform_round,
rho_one_shot, vf_one_shot,
rho_multi, vf_multi
```

所有方法必须读取同一个已保存的 `S0` 和 target initialization。one-shot 在 warm start
后只更新一次；multi-round 只在 `ProtocolSpec.update_epochs()` 返回的边界更新。VF-M 的
continuous `omega`、`theta_hat` 和 `theta_tilde` 均跨 round 保留，Top-B mask 不得覆盖
continuous state。

### 11.3 补齐 checkpoint、恢复和审计

每次 outer update 至少保存：

- resolved config、code/environment metadata 和所有 seeds；
- candidate/holdout/development indices、noise mask 和共享 `S0`；
- target、`theta_hat`、`theta_tilde`、optimizer/scheduler states；
- continuous omega/logits、selected indices、raw/standardized scores；
- 当前 epoch、outer round、inner passes 和 RNG states；
- target/selector processed examples、optimizer steps、wall-clock 和 peak memory；
- Section 8 规定的约束、inner stability、selection 和 accuracy metrics。

恢复后的下一次 update 必须与不中断运行一致。每个 artifact 目录使用唯一 run ID，禁止
静默覆盖已有 run；失败或中断 run 必须保留状态标记。

### 11.4 实现 RQ3 组件开关

在统一 runner 中加入独立、可审计的 switches：

```text
freeze_target
freeze_inner
current_target_as_hat
freeze_joint_comparator
holdout_only_il
batch_local_scoring
reset_omega
delayed_rounding
omega.parameterization={projected,sigmoid}
```

每个 RQ3 run 相对 verified VF-M 只改变一个开关。不得通过不同 runner、不同默认 batch
size 或不同训练预算隐式改变其他组件。

### 11.5 MNIST smoke test

先运行 MNIST、10% noise、seed 1、10% retention：

1. `uniform_fixed` 和 `uniform_round`，验证 target pipeline；
2. `rho_one_shot`，验证 frozen IL、large-batch scoring 和 selected-batch update；
3. `vf_one_shot`，验证两条 inner branch、score sign 和一次 omega update；
4. `vf_multi`，只运行两个 outer updates，验证 persistent state 与 resume；
5. 同一配置从 checkpoint 恢复一次，与不中断运行比较。

smoke test 不用于论文比较。完成标准：无 test leakage；所有 omega constraint audit 通过；
每个 candidate 在 global scoring 中恰好出现一次；inner objective、score cosine、Top-B
Jaccard 和 processed-example counts 可复核；resume 后状态一致。

### 11.6 Correctness preflight 与主实验放行

smoke test 后运行 Stage A：

- `alpha_0 in {1,10}`、`p in {0,0.25}`；
- inner passes `{1,5,10,20,50}`；
- projected 与 budget-calibrated sigmoid；
- target batch size 与 inner batch size 分开筛选；
- ideal RHO、converged VF、truncated VF 和 practical RHO 的方向/排名诊断。

只使用 validation/development 结果冻结 `alpha_r` schedule、inner stopping、warm-start
epochs、outer step size 和两个 update intervals。把冻结后的 resolved configs 另存为
`configs/frozen/`，生成 manifest 后才允许执行 Stage B--F。

### 11.7 服务器端最终完成清单

- [ ] Python/PyTorch/CUDA 环境和代码 snapshot 已记录；
- [ ] `pytest` 与 toy gate 全部通过；
- [ ] 六个核心方法使用同一 runner；
- [ ] shared S0、target init、data order 和 noise realization 可审计；
- [ ] one-shot/multi-round 的真实训练轨迹符合 Section 4；
- [ ] VF 的两条 inner branches、`alpha_r` 和 persistent omega 符合 Section 3；
- [ ] faithful RHO 的 BN/Dropout/buffer 行为通过测试；
- [ ] checkpoint resume 与不中断运行一致；
- [ ] MNIST smoke test 通过且无 test leakage；
- [ ] Stage A 通过 Go/No-Go，冻结配置后再生成主实验 manifest；
- [ ] Stage B--F 严格按 gate 顺序运行，不提前展开旧式全矩阵。
