# Frozen analysis plan: single-hypothesis evaluation of the "rising CRP slope" in Design B

> **English translation of `PRESPEC_SAP.md`.** The Chinese original is the authoritative
> version; this translation is provided so that readers who do not read Chinese can follow the
> plan. The identity of the frozen file is recorded in `PRESPEC_SAP.sha256`.

**Frozen at**: 2026-11-08 09:10 GMT+8 (written before `prespec_slope.py` was first run)
**Nature**: an **internally frozen, post-hoc-informed single-hypothesis evaluation**; **not a
preregistration and not an a-priori hypothesis test**
**Corresponding script**: `prespec_slope.py` → `out/PRESPEC_RESULTS.txt`, `out/prespec_coefs.csv`
**Prior results**: `RESULTS_compare_designs.md` §6.1; `out/SENS_REPL.txt` §C

---

## 0. Honesty statement (to be carried with any report of these results)

The hypothesis evaluated here **was formed after inspecting exploratory results on the same
ELSA data**:

| Exploratory result already seen | Value |
|---|---|
| Design B / CRP / ADL / M2 `time:slope_z` | β = +0.0568, SE = 0.0301, P = 0.0592 |
| CRP > 10 retained (n = 679) | β = +0.0595, P = 0.0216 |
| Disability-worsening event (n = 446, 104 events) | OR = 1.337, P = 0.0214 |
| IPW | β = +0.0651, P = 0.0602 |
| Worst-case imputation | β = +0.0272, P = 0.2612 |
| CHARLS waves 4–5 | β = +0.0533, P = 0.1193 |
| BH-FDR across the 24 primary tests | q = 0.5803 (0 significant) |

Therefore:

1. No P < 0.05 from this layer may be described as "pre-specified", "a priori" or
   "confirmatory".
2. The only purpose of this layer is to decide, **under a frozen single hypothesis, a frozen
   model and frozen criteria**, whether the exploratory signal merits promotion to a main
   conclusion. It cannot manufacture confirmatory evidence.
3. Even if every criterion passes, the conclusion must still be labelled
   **hypothesis-generating, requiring replication in an independent cohort**.
4. If the criteria do not pass, the conclusion is **not confirmed**, and all values are
   reported as they stand, without selecting specifications by P value.

---

## 1. Population and data (frozen)

- Cohort: `out/cohort_designB.csv`, **n = 610** = ELSA wave-2 self-reported doctor-diagnosed
  arthritis ∩ valid CRP at each of waves 2, 4 and 6.
- Exposure window: 2004/05 (t = 0), 2008/09 (t = 1), 2012/13 (t = 2); **the fourth draw
  (2016–19) is excluded** (`w8nurse ⊂ w89nurse`: same specimen, and concurrent with the
  outcome).
- Outcome window: ADL at wave 7 (2014/15, time = 0), wave 8 (2016/17, time = 1), wave 9
  (2018/19, time = 2); primary sample = at least one non-missing ADL (**expected 575 persons /
  1,521 observations**).
- Cleaning (frozen, inherited from `finalize.py`): negative sentinel codes are removed in full
  (−1/−11/−6/−2 …) and `CRP > 10 mg/L` is removed (acute infection). `logcrp = ln(CRP)`.
- Mortality: **no mortality outcome** (0 events in all three cohorts); death enters only as an
  informative-dropout sensitivity analysis.

## 2. Exposure parameterisation (frozen)

Within-person OLS over three equally spaced occasions: `logcrp_it = b0_i + b1_i · t + e`,
t = 0/1/2.

| Symbol | Definition | Unit |
|---|---|---|
| `slope_y` | `b1_i` | log(mg/L) per 4 years |
| `mean_y` | mean of the three occasions (long-term inflammatory level) | log(mg/L) |
| `base_y` | observed value at t = 0 (baseline level) | log(mg/L) |

All three are z-standardised **within the 610-person cohort** (to match the existing main
analysis and allow the reproducibility check).

**Key premise: with three equally spaced occasions, `mean_y ≈ base_y + slope_y`, so level and
change are mathematically coupled.** This layer therefore adds a mandatory **orthogonalised
parameterisation** as a diagnostic:

- `slope_orth` = the person-level residual of `slope_y` regressed on `mean_y + base_y`,
  z-standardised to `slope_orth_z`.
- This represents the change component that is independent of the long-term and baseline level.
- **Criterion (pre-set)**: if the `time ×` coefficient on `slope_orth_z` changes sign or has
  P > 0.10, the signal is judged inseparable from the inflammatory level and cannot be called
  an independent rising-slope effect.

## 3. Single primary hypothesis and single primary test (frozen)

**H1 (direction: β > 0)**: among people with established arthritis, a faster rise in CRP from
wave 2 to wave 6 is associated with a faster increase in ADL disability from wave 7 to wave 9.

**The only primary model** (identical to the existing M2, so that the check reproduces it):

```
y ~ time * mean_z + time * slope_z
    + age_w2 + female + educl + adl6 + bmi_w2 + smoken_w2 + n_comorb
```

- Estimation: marginal linear model with **standard errors clustered on `mergeid`** (the
  random-intercept mixed model did not converge because the covariance was singular; see
  `RESULTS_compare_designs.md` §4.1).
- The only primary term: `time:slope_z`, **two-sided α = 0.05** (no one-sided test).
- One primary test → **no multiplicity correction**; every other analysis is labelled secondary
  or sensitivity and is subjected to BH-FDR as a family.
- Missing data: covariates imputed at the median; outcome analysed as available cases (no
  imputation in the primary analysis).
- Fixed seed `20261108` (bootstrap / permutation / split-half).

**Reproducibility gate (pre-set)**: the primary β and P must agree with the existing
`β = +0.0568 / P = 0.0592` within ±0.0005 / ±0.002; otherwise this entire layer is void and the
data are checked first.

## 4. Pre-specified secondary and sensitivity analyses (all to be reported, none selected)

| ID | Analysis | Pre-specified purpose |
|---|---|---|
| S1 | Binary "rising" group: `rising = slope_y > 0`, `y ~ time*rising + time*mean_z + COV`; sensitivity at 0/±0.10 log thresholds | put "rising" on a clinically interpretable contrast |
| S2 | Observed w7→w9 ADL change by tertile of `slope_y`, with a linear trend test | dose–response |
| S3 | Orthogonalised `slope_orth_z` in the primary model + VIF and correlation matrix for `mean_z`, `slope_z`, `base_z` | identifiability of level vs change |
| S4 | Slope estimators: three-point Theil–Sen median slope, two-point difference `(y6 − y2)/2`, raw CRP scale (not log) | dependence on the estimator |
| S5 | Extended covariates: + `cesd_w2` (depression) + `arth_age` (age at arthritis diagnosis) + `adl_w2` | unmeasured confounding |
| S6 | `CRP > 10` retained (n ≈ 679); no adjustment for `adl6` | cleaning / over-adjustment |
| S7 | IPW (probability of observation at wave 9) + worst-case imputation (LOCF + 90th percentile of the observed increment) | informative dropout |
| S8 | Logistic model for the binary worsening event `adl9 > adl6` (OR) | outcome parameterisation |
| S9 | Cluster bootstrap (B = 1,000 resampling persons), exposure permutation test (B = 1,000, two-sided), split-half replication with a fixed seed | null calibration / stability |
| S10 | Power: observed power, minimum detectable effect (80%, α = 0.05), sample size required for 80% | interpretability of a negative result |
| S11 | CHARLS replication with the same specification (2011→2015 ΔCRP_z → w4→w5 ADL slope, cluster-robust) | cross-cohort direction consistency |

## 5. Pre-specified bias calibration (negative controls and temporal gradient)

| ID | Test | What a non-null result would mean |
|---|---|---|
| NC1 | `adl_w2` (start of the exposure window; before or concurrent with the first draw) ~ `slope_z` + covariates | the rising group was already more disabled → baseline health selection |
| NC2 | Balance of `slope_z` against wave-2 age, sex, education, birth year, height and CES-D (each and jointly, F test) | slope is affected by sociodemographic selection |
| NC3 | **Temporal gradient**: coefficient of `ΔADL(w2→w6)` (concurrent with exposure) ~ `slope_z` versus the coefficient of `ΔADL(w7→w9)` | if concurrent ≥ prospective, the association arises concurrently or by reverse causation rather than as prediction |

NC3 is not a strict negative control (its window overlaps the exposure); it is **pre-specified
as a temporal-gradient test, not a null test**, with a fixed reading rule: if the concurrent
coefficient is ≥ the prospective coefficient, the criterion FAILS.

## 6. Success criteria (frozen before running; all must hold for "confirmed")

| ID | Criterion | Threshold |
|---|---|---|
| **C1** | Primary `time:slope_z` | two-sided P < 0.05 and β > 0 |
| **C2** | S4/S5/S6/S7/S8 all point in the same direction (IPW and worst-case each count as one) | 6/6 without a sign change |
| **C3** | Null calibration | bootstrap P(β > 0) ≥ 0.975 **and** two-sided permutation P < 0.05 |
| **C4** | Negative controls | NC1 P > 0.05, NC2 joint P > 0.05, NC3 concurrent < prospective |
| **C5** | Orthogonalisation | `time:slope_orth_z` β > 0 and P < 0.10 |
| **C6** | Cross-cohort | CHARLS waves 4–5 `time:dcrp_z` β > 0 (significance not required) |
| **C7** | Power | report observed power and MDE; if C1 fails and power < 0.80, write "inconclusive / underpowered" rather than "no effect" |

**Prior expectation**: C1 is expected to FAIL (P = 0.0592 was already known). This layer is
**not** intended to turn it into a PASS; if any criterion passes only because its definition was
relaxed, that must be flagged as a post-hoc change and does not count towards "confirmed".

## 7. Pre-set verdict levels

- **C1–C6 all PASS** → "internally consistent but hypothesis-generating"; independent external
  confirmation is still required, and the result must not be written as confirmatory.
- **C1 FAIL, most others PASS** → "suggestive but not confirmed"; may be reported as an
  exploratory secondary finding, not as a main conclusion.
- **Any C4 FAIL, or S3 showing inseparability** → "likely explained by baseline health or
  mathematical coupling"; recommend abandoning this line.
- **Any negative control significant** → the main effect must not be given a causal reading.

## 8. Out of scope for this layer (to prevent scope drift)

- No new outcomes (mortality, pain, frailty index, hospitalisation); no switch of cohort
  (designs A and C are not rerun).
- No new latent-class growth-mixture classification (the four-class shape has already been
  refuted).
- No new cleaning rules beyond the CRP > 10 retention explicitly listed in §4-S6.
- No manuscript text; this layer delivers statistical results and the criteria check only.
