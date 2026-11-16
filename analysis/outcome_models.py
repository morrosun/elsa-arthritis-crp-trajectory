# -*- coding: utf-8 -*-
"""Head-to-head outcome models for the three competing designs.

For each design x marker x outcome:
  M0  base             : time + covariates
  M1  single-time      : + time x baseline log-CRP (w2), i.e. Ye-2025-like exposure
  M2  trajectory       : + time x (mean level_z, slope_z)  [continuous trajectory]
  M3  full             : + time x (baseline_z, mean_z, slope_z)
  M4  classes          : + time x trajectory class (K=3, and K=4 sensitivity)
Likelihood-ratio tests M1->M3 and M2->M3 answer whether trajectory adds
information beyond a single baseline measurement.
Design C additionally: w9 single-time ANCOVA (OLS, HC3).
Outputs: out/OUTCOME_RESULTS.txt + out/model_compare.csv
"""
import os
import traceback
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "OUTCOME_RESULTS.txt"), "w", encoding="utf-8")


def w(*a):
    print(*a, file=LOG)
    LOG.flush()


def sec(t):
    w("\n" + "=" * 94)
    w("## " + t)
    w("=" * 94)


long = pd.read_csv(os.path.join(OUT, "exposure_long.csv"), dtype={"mergeid": str})
long["mergeid"] = long.mergeid.astype(str).str.strip()
base = pd.read_csv(os.path.join(OUT, "elsa_base.csv"), dtype={"mergeid": str})
base["mergeid"] = base.mergeid.astype(str).str.strip()

COV = ["age_w2", "female", "educl", "adl6", "bmi_w2", "smoken_w2", "n_comorb"]
DESIGN = {
    "A": dict(waves=[6, 7, 8, 9], exp_t=[0, 1, 2, 3], minw=3,
              tag="CORE >=3 draws -> w6-w9 (concurrent exposure)"),
    "B": dict(waves=[7, 8, 9], exp_t=[0, 1, 2], minw=3,
              tag="CLEAN w2/4/6 -> w7-w9 (prospective)"),
    "C": dict(waves=[7, 8, 9], exp_t=[0, 1, 2, 3], minw=4,
              tag="STRICT 4 draws -> w9 single endpoint"),
}
rows = []


def z(s):
    s = pd.Series(s, dtype=float)
    return (s - s.mean()) / (s.std() + 1e-9)


def prep_cov(s):
    s = s.copy()
    for c in COV:
        if c not in s:
            s[c] = np.nan
        s[c] = pd.to_numeric(s[c], errors="coerce")
        med = s[c].median()
        s[c] = s[c].fillna(med if np.isfinite(med) else 0)
    return s


def indiv_exposure(sub, marker, tvals):
    """Continuous individual trajectory parameters from person-wise OLS."""
    s = sub[(sub[marker].notna()) & (sub.t.isin(tvals))]
    rec = []
    for pid, g in s.groupby("mergeid"):
        y = g[marker].to_numpy(float)
        t = g.t.to_numpy(float)
        if len(y) >= 2 and np.ptp(t) > 0:
            b1, b0 = np.polyfit(t, y, 1)
        else:
            b1, b0 = 0.0, float(y.mean())
        rec.append(dict(mergeid=pid, mean_y=float(y.mean()),
                        slope_y=float(b1), base_y=float(g.loc[g.t == g.t.min(), marker].iloc[0]),
                        n_draws=len(y)))
    return pd.DataFrame(rec)


def make_long(s, waves, yvar):
    rec = []
    for wv in waves:
        d = s[["mergeid"]].copy()
        d["y"] = pd.to_numeric(s["%s%d" % (yvar, wv)], errors="coerce")
        for c in COV:
            d[c] = s[c]
        d["time"] = wv - waves[0]
        rec.append(d)
    d = pd.concat(rec, ignore_index=True)
    return d[d.y.notna()].copy()


def fit_mixed(formula, d, label):
    """Marginal (population-averaged) linear model with person-cluster robust SE.
    MixedLM random-intercept variance collapses to 0 here (within-person change
    is small once baseline ADL is adjusted), causing singular covariance.
    Cluster-robust OLS is the stable, standard alternative."""
    r = smf.ols(formula, d).fit(cov_type="cluster",
                                cov_kwds={"groups": d["mergeid"]})
    k = len(r.params) + 1
    aic = -2 * r.llf + 2 * k
    return r, aic, k


def report_terms(r, terms, lab, meta):
    for t in terms:
        if t not in r.params.index:
            continue
        b, se, p = r.params[t], r.bse[t], r.pvalues[t]
        ci = r.conf_int().loc[t].tolist()
        w("    %-34s beta=%+8.4f  SE=%.4f  95%%CI [%+.4f,%+.4f]  P=%.4g" %
          (t, b, se, ci[0], ci[1], p))
        rows.append(dict(**meta, model=lab, term=t, beta=b, se=se, p=p,
                         ci_lo=ci[0], ci_hi=ci[1]))


def cov_str():
    return " + " + " + ".join(COV)


for dk, cfg in DESIGN.items():
    ids = set(pd.read_csv(os.path.join(OUT, "cohort_design%s.csv" % dk),
                          dtype={"mergeid": str}).mergeid.astype(str).str.strip())
    sec("DESIGN %s  n=%d | %s" % (dk, len(ids), cfg["tag"]))
    s0 = prep_cov(base[base.mergeid.isin(ids)].copy())
    for marker in ["logcrp", "logfib"]:
        mk = "crp" if marker == "logcrp" else "fib"
        ex = indiv_exposure(long[long.mergeid.isin(ids)], marker, cfg["exp_t"])
        cnt = long[(long.mergeid.isin(ids)) & long[marker].notna() &
                   long.t.isin(cfg["exp_t"])].groupby("mergeid").size()
        ex = ex[ex.mergeid.isin(set(cnt[cnt >= cfg["minw"]].index))]
        s = s0.merge(ex, on="mergeid", how="inner")
        s["mean_z"] = z(s.mean_y)
        s["slope_z"] = z(s.slope_y)
        s["base_z"] = z(s.base_y)
        s["n_draws"] = s.n_draws
        w("\n--- marker %s : %d persons with trajectory parameters ---" % (mk.upper(), len(s)))
        w("    mean log-%s %.3f (SD %.3f); slope %.4f/4y (SD %.4f); corr(mean,slope)=%.3f" %
          (mk, s.mean_y.mean(), s.mean_y.std(), s.slope_y.mean(), s.slope_y.std(),
           s[["mean_y", "slope_y"]].corr().iloc[0, 1]))
        # attach trajectory classes
        for K in (3, 4):
            f = os.path.join(OUT, "traj_%s_%s_K%d.csv" % (dk, mk, K))
            if os.path.exists(f):
                tc = pd.read_csv(f, dtype={"mergeid": str})
                tc["mergeid"] = tc.mergeid.astype(str).str.strip()
                s = s.merge(tc[["mergeid", "traj_class"]].rename(
                    columns={"traj_class": "cls%d" % K}), on="mergeid", how="left")
        s["cls3"] = s["cls3"].astype("Int64") if "cls3" in s else np.nan
        s["cls4"] = s["cls4"].astype("Int64") if "cls4" in s else np.nan

        for yvar in ["adl", "iadl"]:
            d = make_long(s, cfg["waves"], yvar)
            d = d[d.mergeid.isin(set(s.mergeid))]
            d["cls3"] = d.mergeid.map(s.set_index("mergeid").cls3)
            d["cls4"] = d.mergeid.map(s.set_index("mergeid").cls4)
            for c in ["mean_z", "slope_z", "base_z"]:
                d[c] = d.mergeid.map(s.set_index("mergeid")[c])
            w("\n  [%s | %s] long rows=%d, persons=%d" %
              (mk.upper(), yvar.upper(), len(d), d.mergeid.nunique()))
            meta = dict(design=dk, marker=mk, outcome=yvar, n_person=d.mergeid.nunique(),
                        n_obs=len(d))
            f0 = "y ~ time" + cov_str()
            f1 = "y ~ time * base_z" + cov_str()
            f2 = "y ~ time * mean_z + time * slope_z" + cov_str()
            f3 = "y ~ time * base_z + time * mean_z + time * slope_z" + cov_str()
            res = {}
            for lab, f in [("M0", f0), ("M1", f1), ("M2", f2), ("M3", f3)]:
                try:
                    r, aic, k = fit_mixed(f, d, lab)
                    res[lab] = (r, aic, k)
                    w("    %s : LL=%9.2f  AIC=%9.2f  (cluster-robust OLS)" %
                      (lab, r.llf, aic))
                    terms = {"M1": ["time:base_z"],
                             "M2": ["time:mean_z", "time:slope_z"],
                             "M3": ["time:base_z", "time:mean_z", "time:slope_z"]}.get(lab, [])
                    report_terms(r, terms, lab, meta)
                    rows.append(dict(**meta, model=lab, term="__AIC__", beta=aic,
                                     se=np.nan, p=np.nan, ci_lo=r.llf, ci_hi=k))
                except Exception as e:
                    w("    %s ERROR %r" % (lab, e))
            # Joint Wald test (cluster-robust) on the two trajectory terms in M3:
            # this is the correct incremental test under robust covariance.
            if "M3" in res:
                r3 = res["M3"][0]
                try:
                    wt = r3.wald_test("time:mean_z = 0, time:slope_z = 0",
                                      scalar=True, use_f=True)
                    w("    Wald joint test (mean_z & slope_z) in M3: F=%.3f  P=%.4g  "
                      "df=(%.0f,%.0f)" % (wt.fvalue, wt.pvalue, wt.df_num, wt.df_denom))
                    rows.append(dict(**meta, model="Wald_traj_2df", term="F_2df",
                                     beta=wt.fvalue, se=np.nan, p=wt.pvalue,
                                     ci_lo=wt.df_num, ci_hi=wt.df_denom))
                except Exception as e:
                    w("    Wald test ERROR %r" % e)
                pb = r3.pvalues.get("time:base_z", np.nan)
                w("    in M3, single-time term time:base_z P=%.4g  "
                  "(is baseline still needed once trajectory is in the model?)" % pb)
            if "M1" in res and "M3" in res:
                w("    AIC: M1=%.2f -> M3=%.2f (delta=%+.2f; negative favours trajectory)" %
                  (res["M1"][1], res["M3"][1], res["M3"][1] - res["M1"][1]))
            # class model
            for K in (3, 4):
                col = "cls%d" % K
                if d[col].isna().all():
                    continue
                dd = d[d[col].notna()].copy()
                dd[col] = dd[col].astype(int).astype(str)
                f = "y ~ time * C(%s, Treatment(reference='1'))" % col + cov_str()
                try:
                    r, aic, k = fit_mixed(f, dd, "M4_K%d" % K)
                    inter = [t for t in r.params.index if ":" in t and "time" in t]
                    w("    M4_K%d : LL=%.2f AIC=%.2f  (n=%d)" %
                      (K, r.llf, aic, dd.mergeid.nunique()))
                    report_terms(r, inter, "M4_K%d" % K, meta)
                    rows.append(dict(**meta, model="M4_K%d" % K, term="__AIC__", beta=aic,
                                     se=np.nan, p=np.nan, ci_lo=r.llf, ci_hi=k))
                except Exception as e:
                    w("    M4_K%d ERROR %r" % (K, e))

    # ---- Design C single-time w9 ANCOVA ----
    if dk == "C":
        sec("DESIGN C : w9 single-time endpoint (ANCOVA, HC3)")
        for marker in ["logcrp", "logfib"]:
            mk = "crp" if marker == "logcrp" else "fib"
            ex = indiv_exposure(long[long.mergeid.isin(ids)], marker, [0, 1, 2, 3])
            s = prep_cov(base[base.mergeid.isin(ids)].copy()).merge(ex, on="mergeid", how="inner")
            s["mean_z"], s["slope_z"], s["base_z"] = z(s.mean_y), z(s.slope_y), z(s.base_y)
            f3 = os.path.join(OUT, "traj_C_%s_K3.csv" % mk)
            if os.path.exists(f3):
                tc = pd.read_csv(f3, dtype={"mergeid": str})
                tc["mergeid"] = tc.mergeid.astype(str).str.strip()
                s = s.merge(tc[["mergeid", "traj_class"]], on="mergeid", how="left")
            for yvar in ["adl", "iadl"]:
                s["y"] = s["%s9" % yvar]
                # identical analytic sample across the three models (AIC comparability)
                d = s[s.y.notna() & s.traj_class.notna()].copy()
                d["traj_class"] = d.traj_class.astype(int).astype(str)
                meta = dict(design="C_w9", marker=mk, outcome=yvar,
                            n_person=len(d), n_obs=len(d))
                for lab, f in [
                    ("single", "y ~ base_z + adl6" + cov_str().replace(" + adl6", "")),
                    ("traj", "y ~ mean_z + slope_z + adl6" + cov_str().replace(" + adl6", "")),
                    ("class", "y ~ C(traj_class, Treatment(reference='1')) + adl6" +
                     cov_str().replace(" + adl6", "")),
                ]:
                    try:
                        r = smf.ols(f, d).fit(cov_type="HC3")
                        w("  [%s|%s|%s] n=%d R2=%.4f AIC=%.2f" %
                          (mk, yvar, lab, len(d), r.rsquared, r.aic))
                        for t in [x for x in r.params.index if x in
                                  ("base_z", "mean_z", "slope_z") or "traj_class" in x]:
                            ci = r.conf_int().loc[t].tolist()
                            w("      %-40s beta=%+8.4f SE=%.4f P=%.4g" %
                              (t, r.params[t], r.bse[t], r.pvalues[t]))
                            rows.append(dict(**meta, model="ANCOVA_" + lab, term=t,
                                             beta=r.params[t], se=r.bse[t],
                                             p=r.pvalues[t], ci_lo=ci[0], ci_hi=ci[1]))
                        rows.append(dict(**meta, model="ANCOVA_" + lab, term="__AIC__",
                                         beta=r.aic, se=np.nan, p=np.nan,
                                         ci_lo=r.rsquared, ci_hi=np.nan))
                    except Exception as e:
                        w("  ERROR %s %r" % (lab, e))

pd.DataFrame(rows).to_csv(os.path.join(OUT, "model_compare.csv"),
                          index=False, encoding="utf-8-sig")
w("\nmodel_compare.csv rows=%d" % len(rows))
LOG.close()
print("done")
