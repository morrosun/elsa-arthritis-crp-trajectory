# 冻结分析计划：Design B 的「CRP 上升斜率」单一假设评估

**冻结时间**：2026-11-08 09:10 GMT+8（在 `prespec_slope.py` 首次运行之前写入）
**性质**：**内部冻结的单一假设评估（internally frozen, post-hoc–informed）**，**不是预注册、不是事前假设检验**
**对应脚本**：`prespec_slope.py` → `out/PRESPEC_RESULTS.txt`、`out/prespec_coefs.csv`
**前置结果**：`RESULTS_compare_designs.md` §6.1；`out/SENS_REPL.txt` §C

---

## 0. 诚实性声明（必须随结果一同写进论文）

本计划所检验的假设，**来自对同一批 ELSA 数据探索性结果的检视**：

| 已看到的探索性结果 | 数值 |
|---|---|
| Design B / CRP / ADL / M2 `time:slope_z` | β = +0.0568, SE = 0.0301, P = 0.0592 |
| CRP>10 保留（n=679） | β = +0.0595, P = 0.0216 |
| 失能恶化事件（n=446, 104 events） | OR = 1.337, P = 0.0214 |
| IPW | β = +0.0651, P = 0.0602 |
| 最坏情景填补 | β = +0.0272, P = 0.2612 |
| CHARLS w4–w5 | β = +0.0533, P = 0.1193 |
| BH-FDR（24 主检验） | q = 0.5803（0 项显著） |

因此：

1. **不能**把本层任何 P<0.05 表述为 "pre-specified / a priori / confirmatory"。
2. 本层的唯一目的是：**在冻结的单一假设、冻结的模型与冻结的判据下，判定这条探索性信号是否值得升级为论文主结论**；它不能制造确认性证据。
3. 即使全部判据通过，结论仍须标注为 **hypothesis-generating, requiring replication in an independent cohort**。
4. 若判据未通过，结论即为 **not confirmed**，并如实报告全部数值，不得按 P 值挑选子设定。

---

## 1. 人群与数据（冻结）

- 队列：`out/cohort_designB.csv`，**n = 610**＝ELSA wave 2 自报确诊关节炎 ∩ w2/w4/w6 三次有效 CRP 采血。
- 暴露窗口：2004/05 (t=0)、2008/09 (t=1)、2012/13 (t=2)；**不含 2016–19 第 4 次采血**（`w8nurse ⊂ w89nurse`，同一次采血且结局同期）。
- 结局窗口：ADL w7 (2014/15, time=0)、w8 (2016/17, time=1)、w9 (2018/19, time=2)；主分析样本 = 至少 1 次非缺失 ADL（预期 **575 人 / 1,521 obs**）。
- 清洗（冻结，沿用 `finalize.py`）：负哨兵码**全部**剔除（−1/−11/−6/−2 等），`CRP > 10 mg/L` 剔除（急性感染）。`logcrp = ln(CRP)`。
- 死亡：**不设死亡结局**（三个队列 0 事件）；仅作 informative dropout 敏感性。

## 2. 暴露参数化（冻结）

个体内三点等距 OLS：`logcrp_it = b0_i + b1_i · t + e`，t = 0/1/2。

| 符号 | 定义 | 单位 |
|---|---|---|
| `slope_y` | `b1_i` | log(mg/L) / 4 年 |
| `mean_y` | 三点均值（长期炎症水平） | log(mg/L) |
| `base_y` | t=0 观测值（基线水平） | log(mg/L) |

三者均在**610 人队列内** z 标准化（与主分析既有结果保持一致，以便复现校验）。

**关键预设：三点等距设计下 `mean_y ≈ base_y + slope_y`，水平与变化存在数学耦合。** 因此本层强制加入一个**正交化参数化**作为判别工具：

- `slope_orth` = `slope_y` 对 `mean_y + base_y` 回归的**残差**（person-level），z 化 → `slope_orth_z`。
- 这代表"与长期水平/基线水平无关的纯变化成分"。
- **判据（预设）**：若 `slope_orth_z` 的 `time×` 系数变号或 P>0.10，则判定「信号不可与炎症水平分离，不能称为独立的上升斜率效应」。

## 3. 单一主假设与单一主检验（冻结）

**H1（方向：β>0）**：在已确诊关节炎者中，w2→w6 的 CRP 上升越快，w7→w9 的 ADL 失能加速越快。

**唯一 Primary 模型**（与既有 M2 完全一致，用于复现校验）：

```
y ~ time * mean_z + time * slope_z
    + age_w2 + female + educl + adl6 + bmi_w2 + smoken_w2 + n_comorb
```

- 估计：边际线性模型 + **按 `mergeid` 聚类的稳健 SE**（随机截距混合模型因协方差奇异不收敛，见 `RESULTS_compare_designs.md` §4.1）。
- 唯一 primary 项：`time:slope_z`，**双侧 α = 0.05**（不使用单侧检验）。
- 唯一 primary 检验 → **不做多重比较校正**；所有其余分析一律标为 secondary / sensitivity，并统一做 BH-FDR。
- 缺失：协变量中位数填补；结局用 available-case（不插补为主分析）。
- 固定随机种子 `20261108`（bootstrap / permutation / split-half）。

**复现性门槛（预设）**：primary β 与 P 必须与既有 `β=+0.0568 / P=0.0592` 在 ±0.0005 / ±0.002 内一致，否则本层全部结果作废、先查数据。

## 4. 预设的 secondary / sensitivity 清单（全部要报告，不挑选）

| 编号 | 分析 | 预设用途 |
|---|---|---|
| S1 | 二分类「上升组」：`rising = slope_y > 0`，`y ~ time*rising + time*mean_z + COV`；并给 0/±0.10 log 两个阈值的敏感性 | 把"上升"落到临床可解释的对比 |
| S2 | `slope_y` 三分位组的 w7→w9 ADL 实际变化 + 线性趋势检验 | 剂量—反应 |
| S3 | 正交化 `slope_orth_z` 主模型 + `mean_z / slope_z / base_z` 的 VIF 与相关阵 | 水平 vs 变化的可识别性 |
| S4 | 斜率估计法：Theil–Sen 三点中位斜率、`(y6−y2)/2` 两点差、原始 CRP 尺度（非 log） | 估计方式依赖 |
| S5 | 扩展协变量：+ `cesd_w2`（抑郁）+ `arth_age`（关节炎诊断年龄）+ `adl_w2` | 未测混杂 |
| S6 | `CRP>10` 保留（n≈679）；不调整 `adl6` | 清洗 / 过度调整 |
| S7 | IPW（w9 观测概率）+ 最坏情景填补（LOCF + 观察到增量的 90 分位） | informative dropout |
| S8 | 二分类恶化事件 `adl9 > adl6` 的 logistic（OR） | 结局参数化 |
| S9 | cluster bootstrap（按人重抽 B=1000）、暴露置换检验（B=1000，双侧）、固定种子分半验证 | 零假设校准 / 稳定性 |
| S10 | 效能：observed power、MDE(80%, α=0.05)、达到 80% 所需样本量 | 阴性结果可解释性 |
| S11 | CHARLS 同规格 replication（2011→2015 ΔCRP_z → w4→w5 ADL 斜率，cluster-robust） | 跨库方向一致性 |

## 5. 预设的偏倚校准（负对照 / 时序梯度）

| 编号 | 检验 | 若为「非 null」意味着什么 |
|---|---|---|
| NC1 | `adl_w2`（暴露期起点、早于/同期首次采血）~ `slope_z` + 协变量 | 上升组本来就更虚弱 → 基线健康选择 |
| NC2 | `slope_z` 与 w2 年龄、性别、教育、出生年、身高、CES-D 的平衡（各自 + 联合 F） | 斜率受社会人口学选择影响 |
| NC3 | **时序梯度**：`ΔADL(w2→w6)`（暴露同期窗口）~ `slope_z` 的系数 vs `ΔADL(w7→w9)` 的系数 | 若同期 ≥ 前瞻，则效应主要来自同期/反向因果，而非前瞻预测 |

NC3 不是严格负对照（窗口与暴露重叠），**预设为"时序梯度检验"而非"零效应检验"**，判读标准写死：若同期系数 ≥ 前瞻系数，判 FAIL。

## 6. 成功判据（跑数前冻结；全部满足才算 confirmed）

| 编号 | 判据 | 阈值 |
|---|---|---|
| **C1** | Primary `time:slope_z` | 双侧 P < 0.05 且 β > 0 |
| **C2** | S4/S5/S6/S7/S8 全部符号为正（IPW、最坏填补各算一项） | 6/6 不翻符号 |
| **C3** | 零假设校准 | bootstrap P(β>0) ≥ 0.975 **且** 置换检验双侧 P < 0.05 |
| **C4** | 负对照 | NC1 P>0.05、NC2 联合 P>0.05、NC3 同期系数 < 前瞻系数 |
| **C5** | 正交化 | `time:slope_orth_z` β>0 且 P<0.10 |
| **C6** | 跨库 | CHARLS w4–w5 `time:dcrp_z` β>0（不要求显著） |
| **C7** | 效能 | 报告 observed power 与 MDE；若 C1 FAIL 且 power<0.80，结论写「inconclusive / underpowered」而非「no effect」 |

**先验预期**：C1 的期望是 FAIL（已见 P=0.0592）。本层**不是**为了把它变成 PASS；若任何判据因放宽定义而 PASS，必须在报告中标注为事后调整、不计入 confirmed。

## 7. 判定等级（预设）

- **C1–C6 全 PASS** → "internally consistent but hypothesis-generating"；仍须外部独立队列确认，不得写成 confirmatory。
- **C1 FAIL，其余多数 PASS** → "suggestive but not confirmed"；可作为探索性次要发现写入，不作为主结论。
- **C4 任一 FAIL 或 S3 显示不可分离** → "likely explained by baseline health / mathematical coupling"；建议放弃该主线。
- **任何负对照显著** → 主效应不得作因果解释。

## 8. 不在本层范围内（避免范围漂移）

- 不新增结局（死亡、疼痛、衰弱指数、住院）；不更换队列（A / C 不重跑）。
- 不做 LCGMM 新分类（已证伪四类形态）。
- 不引入新的数据清洗规则（除 §4-S6 明确列出的 CRP>10 保留）。
- 不组装论文正文；本层只交付统计结果与判据核对。
