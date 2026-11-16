# Repeated inflammatory measurements and disability change in arthritis (ELSA)

Analysis code for the manuscript:

> **Repeated CRP and fibrinogen measurements add no reliable information beyond a single
> baseline measurement for disability change in adults with arthritis: a negative
> re-appraisal in ELSA**

The study re-extracts high-sensitivity C-reactive protein (CRP) and fibrinogen from five
ELSA nurse-visit tables (2004–2019), keeps participants who reported doctor-diagnosed
arthritis at wave 2 and had valid CRP at waves 2, 4 and 6, and asks whether a three-occasion
inflammatory trajectory predicts the **rate of change** in ADL/IADL disability at waves 7–9
better than one baseline measurement. Disability is the sole outcome. The answer is no: the
apparent "rising" trajectory cannot be separated from a lower baseline level, regression to
the mean or measurement error, and the increment over a single measurement is not
detectable.

This is a **code archive**, not a data package. Every coefficient and figure is regenerated
from the scripts and the licensed source data; no result table and no respondent-level record
is stored here.

---

## 1. What the repository contains

`analysis/` — 17 Python scripts, grouped by role.

| Group | Scripts | Role |
|---|---|---|
| Data build | `feasibility_01.py`, `feasibility_02.py` | verify the five nurse-visit asset, extract the four blood-draw occasions into a long table, build the person–wave ADL/IADL table, align CHARLS blood IDs |
| Cohort construction | `build_analysis.py`, `finalize.py` | covariates and outcomes in one row per person, the three competing design cohorts (A/B/C), the CORE/STRICT/Option-B cohorts, attrition and death-status audit |
| Trajectory models | `traj_models.py` | group-based trajectory models (K = 3, 4; linear, censored normal) for CRP and fibrinogen in designs A/B/C |
| Design comparison | `outcome_models.py` | head-to-head single-baseline / mean-plus-slope / trajectory models, robust Wald tests, ΔAIC |
| Primary analysis | `prespec_slope.py` | the frozen single-hypothesis evaluation of Design B, with S1–S11 secondary analyses and NC1–NC3 negative controls |
| Sensitivity | `sensitivity_replication.py`, `death_sensitivity.py`, `drug_sensitivity.py`, `drug_class_sensitivity.py`, `addendum.py` | multiplicity and split-half, CHARLS direction consistency, upper-bound and δ–β handling of unknown vital status, wave-6 medication as covariate and by BNF class, survivorship addendum |
| Checks and figures | `check_theil_degeneracy.py`, `math_audit2.py`, `verify_prespec.py`, `make_brief_figures.py`, `make_supp_figure.py` | algebraic identities behind the slope–baseline correlation, second-pass re-fit of the primary model, Figures 1–3 and Figure S1 |
They contribute no estimate to the manuscript, and the reported result is null, so selection
on event counts cannot explain it.

## 2. What the repository deliberately does not contain

- **Respondent-level data.** ELSA is distributed by the UK Data Service under an End User
  Licence, and CHARLS by Peking University under its own data-use agreement. Neither may be
  redistributed. `data/`, `out/`, `*.csv`, `*.rds` and `*.dta` are git-ignored.
- **Derived result tables and logs.** `out/` is created at run time.
- **Manuscript, supplement and report assembly code**, including the HTML builders and the
  reference-fetching and renumbering scripts. Those files embed the unpublished text.
- **Internal review audits** that re-derive manuscript numbers from hard-coded expected
  values. They are quality control on the text, not on the analysis.

## 3. Data and access

| Input | Source | Where the scripts expect it |
|---|---|---|
| ELSA harmonised file (`elsa.RData`) and the five nurse-visit tables | UK Data Service, study SN 5050 (End User Licence) | `ELSA_RDATA` (default `<repo>/data/AGING_DB_R`); the nurse tables are read through PostgreSQL |
| ELSA harmonised variables (`gh_elsa_p1` … `gh_elsa_p5`) | as above, loaded into PostgreSQL | database `elsa` |
| ELSA raw nurse-visit tables (`rw_w2nurse_p1`, `rw_w4nurse_p1`, `rw_w6nurse_p1`, `rw_w8nurse_p1`, `rw_w89nurse_p1`) | as above, loaded into PostgreSQL | database `elsa_raw` |
| CHARLS 2011 and 2015 blood assays and waves 4–5 ADL | Peking University, CHARLS | database `charls` |

`elsa_raw`-style table names are those of the UKDA release; column names beginning `r{w}`
are wave-prefixed ELSA harmonised variables (`r2adla`, `r6adla`, …). ADL is scored 0–5 and
IADL 0–3; the scripts treat both as bounded counts.

**Mortality linkage.** The locally held ELSA release contains no death linkage after 2012, and
the linkage itself is controlled data available only through UK LLC to UK-based institutions.
The analysis therefore does not model mortality; it reports an upper-bound and a δ–β
sensitivity analysis for unknown vital status instead. No script here attempts to recover
post-2012 deaths.

## 4. Environment

Tested with Python 3.13 on Windows. Install with `pip install -r requirements.txt`.

| Variable | Default | Meaning |
|---|---|---|
| `ELSA_RDATA` | `<repo>/data/AGING_DB_R` | directory holding `elsa.RData` |
| `ELSA_DB_HOST` | `localhost` | PostgreSQL host |
| `ELSA_DB_PORT` | `5432` | PostgreSQL port |
| `ELSA_DB_USER` | `postgres` | PostgreSQL user |
| `ELSA_DB_PASSWORD` | *(empty)* | PostgreSQL password — **no credential is stored in this repository** |

All output goes to `<repo>/out`, which the scripts create. The repository root is derived from
the script location, so scripts can be run from any working directory.

## 5. Reproducing the analysis

```bash
mkdir -p out
python analysis/feasibility_01.py         # exposure long table from the nurse-visit tables
python analysis/feasibility_02.py         # person-wave ADL/IADL table
python analysis/build_analysis.py         # covariates/outcomes, design A/B/C cohorts
python analysis/finalize.py               # CORE/STRICT/Option-B cohorts, attrition audit
python analysis/traj_models.py            # GBTM K=3,4 for CRP and fibrinogen
python analysis/outcome_models.py         # design comparison, model_compare.csv
python analysis/prespec_slope.py          # frozen primary analysis + S1-S11 + NC1-NC3
python analysis/sensitivity_replication.py
python analysis/death_sensitivity.py
python analysis/drug_sensitivity.py
python analysis/drug_class_sensitivity.py
python analysis/addendum.py
python analysis/make_brief_figures.py     # Figures 1-3
python analysis/make_supp_figure.py       # Figure S1
```

Later scripts consume the CSV/TXT files written into `out/` by earlier ones. `prespec_slope.py`
refuses to start if the frozen plan (`PRESPEC_SAP.md`) is absent, and logs its timestamp.

Two scripts are independent of the pipeline and can be run at any point:
`check_theil_degeneracy.py` (random-triple demonstration that the three-point Theil–Sen slope
equals the first-to-last difference) and `math_audit2.py` (re-derivation of the algebraic
relations used in the Discussion).

## 6. Relationship to the frozen analysis plan

`PRESPEC_SAP.md` is the plan that was written before `prespec_slope.py` was first run, and
`PRESPEC_SAP_EN.md` is an English translation; the Chinese file is the authoritative version.
The plan governs an internal decision — whether the exploratory signal could be promoted to a
main conclusion — and is **not** a preregistration. As stated in the manuscript, the criteria
were not used as a scorecard: the paper reports estimates with 95% confidence intervals.

| Plan item | Where it appears |
|---|---|
| C1 primary test, C5 orthogonalisation | Table 2; Table S9-1 rows 1–2, 6 |
| C2 sign consistency (S4–S8) | Table S9-1 rows 8–15 |
| C4 negative controls NC1, NC3 | Table S9-1 rows 16–18 |
| C6 CHARLS direction consistency | Table S9-1 row 21 |
| S1 binary rising, S2 tertile trend, S3 orthogonalisation, S4 slope estimators, S5 extended covariates, S6 cleaning, S7 dropout, S8 event outcome, S9 split-half, S11 replication | Table S9-1 rows 3–15, 19–22 |
| C3 bootstrap / permutation calibration | computed and written to `out/PRESPEC_RESULTS.txt`; summarised in the run log rather than tabulated |
| NC2 socio-demographic balance | computed in `prespec_slope.py`; not tabulated in the manuscript |
| C7 observed power, minimum detectable effect, required sample size | computed and written to `out/PRESPEC_RESULTS.txt`; **deliberately not reported** |

**Deviation from the plan, stated explicitly.** The plan listed observed power and the minimum
detectable effect as criterion C7. They are computed, but the manuscript does not report them:
post hoc power is a monotone transformation of the observed P value and adds nothing to the
confidence interval (Hoenig & Heisey, *Am Stat* 2001;55:19–24). Methods therefore say that
power calculations based on observed effects are not reported, not that they were never run.
This is a reporting decision taken after the plan was written; no analysis was changed.

## 7. Determinism

- Global seed `20261108` for bootstrap, permutation and split-half resampling.
- GBTM: eight random starts per K with `numpy.random.default_rng(11)`; KMeans initialisation
  seeded at `11 + start`; separate seeds for the K = 3 and K = 4 fits.
- No network access is used by any script in `analysis/`.

Exact numeric reproduction of the trajectory classes requires the same `scikit-learn` and
`numpy` versions as in `requirements.txt`; the coefficient models themselves are deterministic.

## 8. Figures

`make_brief_figures.py` renders Figures 1–3 and Tables 1, 1B and 2 from outputs already on disk.
Its header states the rule it follows: every number in it is transcribed from an audited output
file (`out/FINAL_AUDIT.txt`, `out/model_compare.csv`, `out/prespec_coefs.csv`), and it performs
no new estimation. Figure 2B reads the coefficient table directly; the cohort-flow counts in
Figure 1B and the P values and ΔAIC values annotated on Figure 3 are layout constants taken
from those same files. Full-precision values are in `out/*.csv`.

## 9. Verification scripts

- `verify_prespec.py` re-fits the primary model from the built tables and checks the
  coefficient, standard error and P value against the frozen values within a stated tolerance.
  This is a reproducibility check on the frozen result, not an independent replication.
- `check_theil_degeneracy.py` shows numerically that for three equally spaced occasions the
  Theil–Sen slope equals `(y2 − y0) / 2` to machine precision.
- `math_audit2.py` re-derives the slope–baseline correlation identity and the difference-score
  reliability formula used in the Discussion.

## 10. Caveats a reader should know

1. The scripts require licensed data and a local PostgreSQL instance; they cannot be run from
   a fresh clone alone.
2. Git does not preserve file modification times, so the timestamp that `prespec_slope.py`
   prints for `PRESPEC_SAP.md` reflects checkout, not authorship. The identity of the frozen
   plan is the SHA-256 recorded in `PRESPEC_SAP.sha256`; the freeze date is stated in the file
   header (2026-11-08 09:10 GMT+8) and in the manuscript's supplement.
3. All analyses are observational. No script implements a causal-inference design.

## 11. License and citation

Code released under the MIT License (see `LICENSE`). The source data remain under their own
agreements and are not covered by this licence.

If you use this code, cite the manuscript and the archived release (DOI to be added at
archiving).
