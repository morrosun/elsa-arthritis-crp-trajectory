# -*- coding: utf-8 -*-
"""Sensitivity + replication + multiplicity, to decide which design to keep.

A. Death handling: (1) delete as outcome [0 events, already established]
                   (2) informative-dropout sensitivity: IPW + worst-case imputation
B. Sensitivity: no baseline-ADL adjustment; CRP>10 kept; disability-worsening event
C. Multiplicity: BH-FDR over the 24 primary trajectory tests
D. CHARLS replication: delta-CRP (2011->2015) -> ADL/IADL slope (w3-w5) and w5 endpoint
Outputs: out/SENS_REPL.txt
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
LOG = open(os.path.join(OUT, "SENS_REPL.txt"), "w", encoding="utf-8")


def w(*a):
    print(*a, file=LOG)
    LOG.flush()


def sec(t):
    w("\n" + "=" * 94)
    w("## " + t)
    w("=" * 94)


long = pd.read_csv(os.path.join(OUT, "exposure_long.csv"), dtype={"mergeid": str})
long["mergeid"] = long.mergeid.astype(str).str.strip()
long["logcrp_all"] = np.log(pd.to_numeric(long.crp_no10, errors="coerce"))
base = pd.read_csv(os.path.join(OUT, "elsa_base.csv"), dtype={"mergeid": str})
base["mergeid"] = base.mergeid.astype(str).str.strip()
mc = pd.read_csv(os.path.join(OUT, "model_compare.csv"))

COV = ["age_w2", "female", "educl", "adl6", "bmi_w2", "smoken_w2", "n_comorb"]
DESIGN = {"A": dict(waves=[6, 7, 8, 9], exp_t=[0, 1, 2, 3]),
          "B": dict(waves=[7, 8, 9], exp_t=[0, 1, 2]),
          "C": dict(waves=[7, 8, 9], exp_t=[0, 1, 2, 3])}


def z(s):
    s = pd.Series(s, dtype=float)
    return (s - s.mean()) / (s.std() + 1e-9)


def prep(s):
    s = s.copy()
    for c in COV:
        s[c] = pd.to_numeric(s.get(c), errors="coerce")
        s[c] = s[c].fillna(s[c].median() if s[c].notna().any() else 0)
    return s


def indiv(sub, marker, tvals):
    s = sub[(sub[marker].notna()) & (sub.t.isin(tvals))]
    rec = []
    for pid, g in s.groupby("mergeid"):
        y, t = g[marker].to_numpy(float), g.t.to_numpy(float)
        b1, b0 = (np.polyfit(t, y, 1) if len(y) >= 2 and np.ptp(t) > 0
                  else (0.0, float(y.mean())))
        rec.append(dict(mergeid=pid, mean_y=float(y.mean()), slope_y=float(b1),
                        base_y=float(g.loc[g.t == g.t.min(), marker].iloc[0]),
                        n_draws=len(y)))
    return pd.DataFrame(rec)


def cohort(dk, marker="logcrp", keep10=False):
    ids = set(pd.read_csv(os.path.join(OUT, "cohort_design%s.csv" % dk),
                          dtype={"mergeid": str}).mergeid.astype(str).str.strip())
    if keep10:
        # rebuild the cohort from ALL w2 arthritis cases, allowing CRP>10
        v = long[(long.crp_no10.notna()) & (long.t.isin(DESIGN[dk]["exp_t"]))]
        cnt = v.groupby("mergeid").size()
        minw = 4 if dk == "C" else 3
        arth2 = set(base.loc[base.arth2 == 1, "mergeid"].astype(str))
        ids = arth2 & set(cnt[cnt >= minw].index)
        marker = "logcrp_all"
    cfg = DESIGN[dk]
    s = prep(base[base.mergeid.isin(ids)].copy())
    ex = indiv(long[long.mergeid.isin(ids)], marker, cfg["exp_t"])
    s = s.merge(ex, on="mergeid", how="inner")
    for c, src in [("mean_z", "mean_y"), ("slope_z", "slope_y"), ("base_z", "base_y")]:
        s[c] = z(s[src])
    return s, cfg


def make_long(s, waves, yvar):
    rec = []
    for wv in waves:
        d = s[["mergeid"]].copy()
        d["y"] = pd.to_numeric(s["%s%d" % (yvar, wv)], errors="coerce")
        for c in COV:
            d[c] = s[c]
        d["time"] = wv - waves[0]
        d["wave"] = wv
        rec.append(d)
    return pd.concat(rec, ignore_index=True)


def fit(formula, d, weights=None):
    gcol = "mergeid" if "mergeid" in d.columns else "pid"
    if weights is None:
        return smf.ols(formula, d).fit(cov_type="cluster",
                                       cov_kwds={"groups": d[gcol]})
    return smf.wls(formula, d, weights=weights).fit(
        cov_type="cluster", cov_kwds={"groups": d[gcol]})


def show(r, terms, lab):
    for t in terms:
        if t in r.params.index:
            w("    %-46s beta=%+8.4f  SE=%.4f  P=%.4g" %
              (lab + " " + t, r.params[t], r.bse[t], r.pvalues[t]))


# ---------------------------------------------------------------- A. multiplicity
sec("A. Multiplicity: BH-FDR over the 24 primary continuous-trajectory tests")
prim = mc[(mc.model == "M2") & (mc.term.isin(["time:mean_z", "time:slope_z"])) &
          (mc.p.notna())].copy()
w("primary tests found: %d (expect 24 = 3 designs x 2 markers x 2 outcomes x 2 terms)" %
  len(prim))
if len(prim):
    p = prim.p.to_numpy()
    order = np.argsort(p)
    q = np.empty_like(p)
    m = len(p)
    q[order] = np.minimum.accumulate((p[order] * m / (np.arange(m) + 1))[::-1])[::-1]
    q = np.clip(q, 0, 1)
    prim["q_BH"] = q
    prim = prim.sort_values("p")
    w(prim[["design", "marker", "outcome", "term", "beta", "p", "q_BH"]]
      .to_string(index=False, float_format=lambda x: "%.4g" % x))
    w("\n  tests with raw P<0.05 : %d" % (p < 0.05).sum())
    w("  tests with BH q<0.05  : %d" % (q < 0.05).sum())
cls = mc[(mc.model.str.startswith("M4")) & mc.term.str.contains(":", na=False) &
         mc.p.notna()]
w("\nclass-model interaction terms: %d, raw P<0.05: %d, min P=%.4g" %
  (len(cls), (cls.p < 0.05).sum(), cls.p.min() if len(cls) else np.nan))
if len(cls):
    w("  BH over ALL interaction tests (primary + class):")
    allp = np.concatenate([p, cls.p.to_numpy()])
    o = np.argsort(allp)
    qq = np.empty_like(allp)
    qq[o] = np.minimum.accumulate((allp[o] * len(allp) / (np.arange(len(allp)) + 1))[::-1])[::-1]
    w("    n=%d, raw P<0.05=%d, BH q<0.05=%d, min q=%.4g" %
      (len(allp), (allp < 0.05).sum(), (np.clip(qq, 0, 1) < 0.05).sum(),
       np.clip(qq, 0, 1).min()))

# ---------------------------------------------------------------- B. death handling
sec("B. Death handling: (1) delete vs (2) informative-dropout sensitivity")
w("Established fact: 0/822 (0/610, 0/352) confirmed deaths -> a survival endpoint")
w("is impossible. Handling (1) = delete mortality, already used in all main models.")
w("Handling (2) tests whether missing w9 disability (including iw9=9 'unknown vital")
w("status') could bias the trajectory coefficient.\n")
for dk in ["A", "B", "C"]:
    try:
        s, cfg = cohort(dk)
        s["dropout"] = s.adl9.isna().astype(int)
        w("Design %s: n=%d, w9 ADL missing (informative dropout proxy)=%d (%.1f%%)" %
          (dk, len(s), s.dropout.sum(), 100 * s.dropout.mean()))
        # propensity for being observed at w9
        f_ps = "dropout ~ mean_z + slope_z + " + " + ".join(
            [c for c in COV if c != "adl9"])
        psr = smf.logit(f_ps, s).fit(disp=0)
        s["p_obs"] = 1 - psr.predict(s)
        pobs = 1 - s.dropout.mean()
        s["ipw"] = np.clip(pobs / np.clip(s.p_obs, 0.05, 1), 0.2, 5.0)
        w("  IPW stabilized weights: mean=%.3f range=[%.2f,%.2f]" %
          (s.ipw.mean(), s.ipw.min(), s.ipw.max()))
        d = make_long(s, cfg["waves"], "adl")
        d = d[d.y.notna()]
        for c in ["mean_z", "slope_z", "base_z"]:
            d[c] = d.mergeid.map(s.set_index("mergeid")[c])
        d["ipw"] = d.mergeid.map(s.set_index("mergeid").ipw)
        f2 = "y ~ time * mean_z + time * slope_z + " + " + ".join(COV)
        r0 = fit(f2, d)
        show(r0, ["time:mean_z", "time:slope_z"], "  delete(as main)")
        r1 = fit(f2, d, weights=d.ipw)
        show(r1, ["time:mean_z", "time:slope_z"], "  IPW")
        # worst-case imputation for missing w9: LOCF + 90th pct observed increment
        obs_inc = (s.adl9 - s.adl8).dropna()
        worst = obs_inc.quantile(0.90) if len(obs_inc) else 1.0
        s2 = s.copy()
        fill_from = s2.adl8.fillna(s2.adl7).fillna(s2.adl6)
        s2["adl9_worst"] = s2.adl9.fillna(fill_from + worst)
        d2 = make_long(s2.assign(adl9=s2.adl9_worst), cfg["waves"], "adl")
        d2 = d2[d2.y.notna()]
        for c in ["mean_z", "slope_z", "base_z"]:
            d2[c] = d2.mergeid.map(s2.set_index("mergeid")[c])
        r2 = fit(f2, d2)
        show(r2, ["time:mean_z", "time:slope_z"], "  worst-case(+%.2f)" % worst)
    except Exception as e:
        w("  Design %s ERROR %r" % (dk, e))
        w(traceback.format_exc()[-400:])

# ---------------------------------------------------------------- C. other sensitivity
sec("C. Other sensitivity analyses (Design B, CRP, ADL)")
try:
    s, cfg = cohort("B")
    d = make_long(s, [7, 8, 9], "adl")
    d = d[d.y.notna()]
    for c in ["mean_z", "slope_z", "base_z"]:
        d[c] = d.mergeid.map(s.set_index("mergeid")[c])
    base_cov = " + ".join(COV)
    f2 = "y ~ time * mean_z + time * slope_z + " + base_cov
    r = fit(f2, d)
    show(r, ["time:mean_z", "time:slope_z"], "  main (adj baseline ADL)")
    r = fit("y ~ time * mean_z + time * slope_z + " +
            " + ".join([c for c in COV if c != "adl6"]), d)
    show(r, ["time:mean_z", "time:slope_z"], "  no baseline-ADL adj")
    # CRP>10 kept
    s10, _ = cohort("B", keep10=True)
    d10 = make_long(s10, [7, 8, 9], "adl")
    d10 = d10[d10.y.notna()]
    for c in ["mean_z", "slope_z", "base_z"]:
        d10[c] = d10.mergeid.map(s10.set_index("mergeid")[c])
    r = fit(f2, d10)
    w("  [CRP>10 kept] n_person=%d" % d10.mergeid.nunique())
    show(r, ["time:mean_z", "time:slope_z"], "  CRP>10 kept")
    # binary worsening event w6 -> w9
    s["worse"] = (s.adl9 > s.adl6).astype(float)
    s.loc[s.adl9.isna(), "worse"] = np.nan
    ev = s[s.worse.notna()]
    w("  [worsening event ADL9>ADL6] n=%d, events=%d (%.1f%%)" %
      (len(ev), ev.worse.sum(), 100 * ev.worse.mean()))
    lr = smf.logit("worse ~ mean_z + slope_z + " + " + ".join(COV), ev).fit(disp=0)
    for t in ["mean_z", "slope_z"]:
        w("    logistic %-10s OR=%.3f  P=%.4g" %
          (t, np.exp(lr.params[t]), lr.pvalues[t]))
except Exception as e:
    w("ERROR %r" % e)
    w(traceback.format_exc()[-500:])

# ---------------------------------------------------------------- D. CHARLS
sec("D. CHARLS replication (direction consistency)")
try:
    ch = pd.read_csv(os.path.join(OUT, "charls_base.csv"), dtype={"pid": str})
    ch["pid"] = ch.pid.astype(str).str.strip()
    w("CHARLS base rows=%d persons=%d" % (len(ch), ch.pid.nunique()))
    w1 = ch[ch.wave == 1].set_index("pid")
    covcols = ["age2011", "ragender", "raeducl", "bmi", "smoken", "hibpe", "diabe", "hearte"]
    for c in covcols:
        if c not in w1:
            w1[c] = np.nan
        w1[c] = pd.to_numeric(w1[c], errors="coerce")
        w1[c] = w1[c].fillna(w1[c].median() if w1[c].notna().any() else 0)
    w1["adl1"] = pd.to_numeric(w1.adlab_c, errors="coerce")
    w1["iadl1"] = pd.to_numeric(w1.iadl, errors="coerce")
    w1["adl1"] = w1.adl1.fillna(w1.adl1.median())
    w1["iadl1"] = w1.iadl1.fillna(w1.iadl1.median())
    exp = ch[ch.wave == 1][["pid", "dcrp"]].dropna().set_index("pid")
    common = w1.index.intersection(exp.index)
    w("persons with baseline covariates + delta-CRP: %d" % len(common))
    longrows = []
    for wv in [3, 4, 5]:
        d = ch[ch.wave == wv].set_index("pid")
        for yv, ycol in [("adl", "adlab_c"), ("iadl", "iadl")]:
            tmp = pd.DataFrame(index=common)
            tmp["pid"] = common
            tmp["y"] = pd.to_numeric(d[ycol].reindex(common), errors="coerce").values
            tmp["outcome"] = yv
            tmp["time"] = wv - 3
            longrows.append(tmp)
    dd = pd.concat(longrows, ignore_index=True)
    dd["dcrp_z"] = z(dd.pid.map(exp.dcrp))
    for c in covcols + ["adl1", "iadl1"]:
        dd[c] = dd.pid.map(w1[c])
    dd["base_fun"] = np.where(dd.outcome == "adl", dd.adl1, dd.iadl1)
    covs = " + ".join(covcols + ["base_fun"])
    for yv in ["adl", "iadl"]:
        for label, sub in [("w3-w5 slope", dd[(dd.outcome == yv) & dd.y.notna()]),
                           ("w4-w5 prospective only",
                            dd[(dd.outcome == yv) & dd.y.notna() & (dd.time >= 1)])]:
            r = fit("y ~ time * dcrp_z + " + covs, sub)
            w("  [%s | %s] n_person=%d n_obs=%d" %
              (yv.upper(), label, sub.pid.nunique(), len(sub)))
            show(r, ["time:dcrp_z"], "    delta-CRP x time")
        # w5 single endpoint (mirrors Design C)
        e5 = ch[ch.wave == 5].set_index("pid")
        one = pd.DataFrame(index=common)
        one["pid"] = common
        one["y"] = pd.to_numeric(e5["adlab_c" if yv == "adl" else "iadl"]
                                 .reindex(common), errors="coerce").values
        one = one[one.y.notna()].copy()
        one["dcrp_z"] = z(one.pid.map(exp.dcrp))
        for c in covcols + ["adl1", "iadl1"]:
            one[c] = one.pid.map(w1[c])
        one["base_fun"] = np.where(yv == "adl", one.adl1, one.iadl1)
        r = smf.ols("y ~ dcrp_z + " + covs, one).fit(cov_type="HC3")
        w("  [%s | w5 single endpoint] n=%d: delta-CRP beta=%+.4f P=%.4g" %
          (yv.upper(), len(one), r.params["dcrp_z"], r.pvalues["dcrp_z"]))
    # group comparison (tertiles of delta CRP)
    w("\n  delta-CRP tertile groups (2011->2015):")
    tz = pd.qcut(exp.dcrp, 3, labels=["falling", "stable", "rising"],
                 duplicates="drop")
    for g in ["falling", "stable", "rising"]:
        pids = set(exp.index[tz == g])
        sub = dd[(dd.outcome == "adl") & dd.pid.isin(pids) & dd.y.notna()]
        if len(sub):
            w("    %-8s n=%4d  ADL w3=%.3f w5=%.3f" %
              (g, sub.pid.nunique(),
               sub[sub.time == 0].y.mean(), sub[sub.time == 2].y.mean()))
except Exception as e:
    w("ERROR %r" % e)
    w(traceback.format_exc()[-600:])

LOG.close()
print("done")
