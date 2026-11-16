# -*- coding: utf-8 -*-
"""Frozen single-hypothesis evaluation of the Design B "rising CRP slope" signal.

IMPORTANT: the hypothesis was formed AFTER inspecting exploratory results.
See PRESPEC_SAP.md (frozen before this script was first run). This script does
NOT produce confirmatory evidence; it applies a frozen set of success criteria
to decide whether the signal can be promoted to a manuscript main conclusion.

Outputs: out/PRESPEC_RESULTS.txt, out/prespec_coefs.csv,
         out/prespec_bootstrap.csv, out/prespec_perm.csv
"""
import os
import datetime
import time
import traceback
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from patsy import dmatrix

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)
SAP = os.path.join(BASE, "PRESPEC_SAP.md")
SEED = 20261108
rng = np.random.default_rng(SEED)

LOG = open(os.path.join(OUT, "PRESPEC_RESULTS.txt"), "w", encoding="utf-8")
ROWS = []
CRIT = {}


def w(*a):
    print(*a, file=LOG)
    LOG.flush()


def sec(t):
    w("\n" + "=" * 96)
    w("## " + t)
    w("=" * 96)


def rec(section, label, term, beta=np.nan, se=np.nan, p=np.nan,
        ci_lo=np.nan, ci_hi=np.nan, n=np.nan, note=""):
    ROWS.append(dict(section=section, label=label, term=term, beta=beta, se=se,
                     p=p, ci_lo=ci_lo, ci_hi=ci_hi, n=n, note=note))
    w("    %-42s beta=%+9.4f  SE=%8.4f  95%%CI [%+.4f,%+.4f]  P=%.4g  n=%s %s" %
      (label + " | " + term, beta, se, ci_lo, ci_hi, p, n, note))
    return beta


def z(s):
    s = pd.Series(s, dtype=float)
    return (s - s.mean()) / (s.std() + 1e-9)


def fit(formula, d, weights=None, key="mergeid"):
    if weights is None:
        return smf.ols(formula, d).fit(cov_type="cluster",
                                       cov_kwds={"groups": d[key]})
    return smf.wls(formula, d, weights=weights).fit(
        cov_type="cluster", cov_kwds={"groups": d[key]})


def term(r, section, label, t, n, note=""):
    if t not in r.params.index:
        w("    MISSING TERM %s" % t)
        return np.nan, np.nan, np.nan
    ci = r.conf_int().loc[t].tolist()
    rec(section, label, t, r.params[t], r.bse[t], r.pvalues[t],
        ci[0], ci[1], n, note)
    return r.params[t], r.bse[t], r.pvalues[t]


# ---------------------------------------------------------------- 0. freeze audit
sec("0. AUDIT: SAP frozen before this run")
sap_mt = datetime.datetime.fromtimestamp(os.path.getmtime(SAP)) if os.path.exists(SAP) else None
w("  SAP path            : %s" % SAP)
w("  SAP exists          : %s" % os.path.exists(SAP))
w("  SAP last modified   : %s" % (sap_mt.strftime("%Y-%m-%d %H:%M:%S") if sap_mt else "MISSING"))
w("  analysis run start  : %s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
w("  random seed         : %d" % SEED)
if not os.path.exists(SAP):
    w("  !! SAP missing -- refusing to run a 'pre-specified' analysis without a frozen plan")
    raise SystemExit(2)
w("  Status: hypothesis is POST-HOC INFORMED (see SAP section 0). "
  "No result below may be labelled confirmatory.")

# ---------------------------------------------------------------- 1. data
long = pd.read_csv(os.path.join(OUT, "exposure_long.csv"), dtype={"mergeid": str})
long["mergeid"] = long.mergeid.astype(str).str.strip()
long["logcrp_all"] = np.log(pd.to_numeric(long.crp_no10, errors="coerce"))
base = pd.read_csv(os.path.join(OUT, "elsa_base.csv"), dtype={"mergeid": str})
base["mergeid"] = base.mergeid.astype(str).str.strip()

COV = ["age_w2", "female", "educl", "adl6", "bmi_w2", "smoken_w2", "n_comorb"]
COV_NOADL = [c for c in COV if c != "adl6"]
FML = "y ~ time * mean_z + time * slope_z + " + " + ".join(COV)
ids = set(pd.read_csv(os.path.join(OUT, "cohort_designB.csv"),
                      dtype={"mergeid": str}).mergeid.astype(str).str.strip())


def prep(s, extra=None):
    s = s.copy()
    for c in COV + (extra or []):
        if c not in s.columns:
            s[c] = np.nan
        s[c] = pd.to_numeric(s[c], errors="coerce")
        med = s[c].median()
        s[c] = s[c].fillna(med if np.isfinite(med) else 0.0)
    return s


def indiv(sub, marker="logcrp", tvals=(0, 1, 2), prefix=""):
    """Person-wise 3-point exposure parameters.

    NOTE (mathematical): with equally spaced t=0,1,2, the OLS slope is exactly
    (y2-y0)/2 -- the middle draw contributes to the intercept but NOT to slope.
    """
    s = sub[(sub[marker].notna()) & (sub.t.isin(tvals))].copy()
    recs = []
    for pid, g in s.groupby("mergeid"):
        g = g.sort_values("t")
        t = g.t.to_numpy(float)
        yl = g[marker].to_numpy(float)
        if len(yl) < 3 or np.ptp(t) == 0:
            continue
        b1, b0 = np.polyfit(t, yl, 1)
        raw = np.exp(yl)
        b1_raw = np.polyfit(t, raw, 1)[0]
        ts = float(np.median([yl[1] - yl[0], yl[2] - yl[1], (yl[2] - yl[0]) / 2.0]))
        recs.append(dict(mergeid=pid, mean_y=float(yl.mean()), slope_y=float(b1),
                         base_y=float(yl[0]), slope_2pt=(yl[2] - yl[0]) / 2.0,
                         slope_ts=ts, slope_raw=b1_raw, n_draws=len(yl),
                         crp0=float(raw[0]), crp1=float(raw[1]), crp2=float(raw[2])))
    return pd.DataFrame(recs)


def make_long(s, waves, yvar, covs=COV):
    rows = []
    for wv in waves:
        d = s[["mergeid"]].copy()
        d["y"] = pd.to_numeric(s["%s%d" % (yvar, wv)], errors="coerce")
        for c in covs:
            d[c] = s[c]
        d["time"] = wv - waves[0]
        d["wave"] = wv
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    for c in ["mean_z", "slope_z", "base_z", "slope_orth_z", "rising", "rising10",
              "grp", "slope_ts_z", "slope_raw_z", "cesd_w2", "arth_age", "adl_w2"]:
        if c in s.columns:
            d[c] = d.mergeid.map(s.set_index("mergeid")[c])
    return d


def build(keep10=False, extra=None):
    if keep10:
        v = long[(long.logcrp_all.notna()) & (long.t.isin([0, 1, 2]))]
        cnt = v.groupby("mergeid").size()
        arth2 = set(base.loc[base.arth2 == 1, "mergeid"].astype(str))
        cohort = arth2 & set(cnt[cnt >= 3].index)
        marker = "logcrp_all"
    else:
        cohort = ids
        marker = "logcrp"
    ex = indiv(long[long.mergeid.isin(cohort)], marker)
    s = prep(base[base.mergeid.isin(cohort)].copy(), extra=extra).merge(
        ex, on="mergeid", how="inner")
    s = s[s.n_draws == 3].copy()
    for c, src in [("mean_z", "mean_y"), ("slope_z", "slope_y"), ("base_z", "base_y"),
                   ("slope_ts_z", "slope_ts"), ("slope_raw_z", "slope_raw")]:
        s[c] = z(s[src])
    # orthogonalized slope: pure change, independent of level
    Xo = np.column_stack([np.ones(len(s)), s.mean_y.to_numpy(), s.base_y.to_numpy()])
    s["slope_orth_z"] = z(s.slope_y.to_numpy() - Xo @ np.linalg.lstsq(Xo, s.slope_y.to_numpy(), rcond=None)[0])
    s["rising"] = (s.slope_y > 0).astype(int)
    s["rising10"] = (s.slope_y > 0.10).astype(int)
    s["grp"] = pd.qcut(s.slope_y, 3, labels=False, duplicates="drop").astype(float)
    return s


sec("1. FROZEN COHORT & EXPOSURE DESCRIPTION")
s = build()
w("  frozen cohort B (arth2 x 3 valid CRP draws w2/w4/w6): n=%d" % len(s))
w("  slope_y (log mg/L per 4y): mean=%.4f SD=%.4f median=%.4f IQR=[%.4f,%.4f]" %
  (s.slope_y.mean(), s.slope_y.std(), s.slope_y.median(),
   s.slope_y.quantile(.25), s.slope_y.quantile(.75)))
w("  OLS slope == (y6-y2)/2 exactly? max abs diff = %.3g" %
  np.max(np.abs(s.slope_y - s.slope_2pt)))
w("  corr(slope_y, base_y)=%.3f ; corr(slope_y, mean_y)=%.3f ; "
  "corr(slope_orth_z, slope_z)=%.3f" %
  (s.slope_y.corr(s.base_y), s.slope_y.corr(s.mean_y),
   s.slope_orth_z.corr(s.slope_z)))
w("  rising (slope>0): %d (%.1f%%) ; rising >0.10 log/4y: %d (%.1f%%)" %
  (s.rising.sum(), 100 * s.rising.mean(), s.rising10.sum(), 100 * s.rising10.mean()))
for g, lab in [(0, "falling tertile"), (1, "middle tertile"), (2, "rising tertile")]:
    q = s[s.grp == g]
    w("  %-16s n=%3d  CRP w2/w4/w6 geometric mean = %.2f / %.2f / %.2f mg/L ; slope=%+.3f" %
      (lab, len(q), np.exp(np.log(q.crp0).mean()), np.exp(np.log(q.crp1).mean()),
       np.exp(np.log(q.crp2).mean()), q.slope_y.mean()))

d = make_long(s, [7, 8, 9], "adl")
d = d[d.y.notna()].copy()
w("  primary analytic long table: %d obs / %d persons ; ADL w7/w8/w9 mean = %.3f / %.3f / %.3f" %
  (len(d), d.mergeid.nunique(), d[d.wave == 7].y.mean(),
   d[d.wave == 8].y.mean(), d[d.wave == 9].y.mean()))

# ---------------------------------------------------------------- 2. PRIMARY
sec("2. PRIMARY (single frozen test): time x slope_z on ADL w7-w9")
r0 = fit(FML, d)
b0, se0, p0 = term(r0, "PRIMARY", "M2_frozen", "time:slope_z", d.mergeid.nunique())
term(r0, "PRIMARY", "M2_frozen", "time:mean_z", d.mergeid.nunique())
w("    replication check vs previous run (beta=+0.0568, P=0.0592): "
  "delta_beta=%.5f delta_P=%.5f" % (b0 - 0.05681, p0 - 0.05924))
rep_ok = abs(b0 - 0.05681) < 0.0005 and abs(p0 - 0.05924) < 0.002
w("    REPRODUCIBILITY GATE: %s" % ("PASS" if rep_ok else "FAIL -- all downstream results void"))
CRIT["REPRODUCIBILITY"] = rep_ok
CRIT["C1"] = bool(p0 < 0.05 and b0 > 0)
w("    C1 (two-sided P<0.05 and beta>0): %s" % ("PASS" if CRIT["C1"] else "FAIL"))

# ---------------------------------------------------------------- 3. secondary / sensitivity
sec("3. PRE-SPECIFIED SECONDARY & SENSITIVITY ANALYSES")
SIGN = {}

# S1 binary rising
for col, lab in [("rising", "rising>0"), ("rising10", "rising>0.10")]:
    f = "y ~ time * %s + time * mean_z + " % col + " + ".join(COV)
    r = fit(f, d)
    b, _, _ = term(r, "S1_binary", lab, "time:" + col, d.mergeid.nunique())
    SIGN["S1_" + lab] = b

# S2 tertile dose-response
f = "y ~ time * grp + " + " + ".join(COV)
r = fit(f, d)
term(r, "S2_trend", "tertile_linear", "time:grp", d.mergeid.nunique())
w("  observed ADL change by slope tertile (w7 -> w9):")
for g, lab in [(0, "falling"), (1, "middle"), (2, "rising")]:
    sub = d[d.grp == g]
    a = sub[sub.wave == 7].y.mean()
    b = sub[sub.wave == 9].y.mean()
    w("    %-8s n=%3d  ADL w7=%.3f  w9=%.3f  delta=%+.3f" %
      (lab, sub.mergeid.nunique(), a, b, b - a))

# S3 orthogonal slope + collinearity
sec("3b. S3: IS THE SIGNAL A 'CHANGE' EFFECT OR A 'LEVEL' EFFECT?")
f = "y ~ time * mean_z + time * slope_orth_z + " + " + ".join(COV)
r = fit(f, d)
bo, seo, po = term(r, "S3_orthogonal", "orth_slope", "time:slope_orth_z", d.mergeid.nunique())
f3 = "y ~ time * base_z + time * mean_z + time * slope_z + " + " + ".join(COV)
r3 = fit(f3, d)
b3, se3, p3 = term(r3, "S3_orthogonal", "M3_all_three", "time:slope_z", d.mergeid.nunique())
w("    SE inflation when base_z added to mean_z+slope_z: %.4f -> %.4f (x%.2f)" %
  (se0, se3, se3 / se0))
V = s[["mean_z", "slope_z", "base_z", "slope_orth_z"]]
w("    correlation matrix:\n%s" % V.corr().round(3).to_string())
for c in ["mean_z", "slope_z", "base_z"]:
    others = [x for x in ["mean_z", "slope_z", "base_z"] if x != c]
    Xc = np.column_stack([np.ones(len(s))] + [s[o].to_numpy() for o in others])
    yv = s[c].to_numpy()
    resid = yv - Xc @ np.linalg.lstsq(Xc, yv, rcond=None)[0]
    r2 = 1 - resid.var() / yv.var()
    w("    VIF(%s) = %.2f" % (c, 1 / max(1 - r2, 1e-9)))
CRIT["C5"] = bool(bo > 0 and po < 0.10)
w("    C5 (orthogonal slope beta>0 and P<0.10): %s" % ("PASS" if CRIT["C5"] else "FAIL"))
w("    NOTE: corr(slope_z, slope_orth_z)=%.3f -> ~%.0f%% of slope variance is shared with "
  "the level terms; the orthogonal (pure-change) component carries almost no association." %
  (s.slope_z.corr(s.slope_orth_z),
   100 * (1 - s.slope_z.corr(s.slope_orth_z) ** 2)))

# S4 slope estimators
sec("3c. S4/S5/S6/S8: estimator, covariate, cleaning and endpoint sensitivity")
for col, lab in [("slope_ts_z", "TheilSen"), ("slope_raw_z", "raw_CRP_scale")]:
    dd = d.copy()
    dd["slope_z"] = dd[col]
    f = "y ~ time * mean_z + time * slope_z + " + " + ".join(COV)
    r = fit(f, dd)
    b, _, _ = term(r, "S4_estimator", lab, "time:slope_z", dd.mergeid.nunique())
    SIGN["S4_" + lab] = b

# S5 extended covariates
extra = [c for c in ["cesd_w2", "arth_age"] if c in base.columns]
w("  extra covariate columns available: %s" % extra)
if extra:
    d5 = make_long(s, [7, 8, 9], "adl", COV + extra)
    d5 = d5[d5.y.notna()].copy()
    for c in extra:
        d5[c] = pd.to_numeric(d5[c], errors="coerce")
        d5[c] = d5[c].fillna(d5[c].median() if d5[c].notna().any() else 0.0)
    f = FML + " + " + " + ".join(extra)
    r = fit(f, d5)
    b, _, _ = term(r, "S5_covars", "extended", "time:slope_z", d5.mergeid.nunique())
    SIGN["S5_extended"] = b
f = "y ~ time * mean_z + time * slope_z + " + " + ".join(COV_NOADL)
r = fit(f, d)
b, _, _ = term(r, "S6_cleaning", "no_adl6_adjustment", "time:slope_z", d.mergeid.nunique())
SIGN["S6_no_adl6"] = b
s10 = build(keep10=True)
d10 = make_long(s10, [7, 8, 9], "adl")
d10 = d10[d10.y.notna()].copy()
r = fit(FML, d10)
b, _, _ = term(r, "S6_cleaning", "CRP>10_kept", "time:slope_z", d10.mergeid.nunique())
SIGN["S6_CRP10_kept"] = b

# S7 IPW + worst case
try:
    sp = s.copy()
    sp["dropout"] = sp.adl9.isna().astype(int)
    ps = smf.logit("dropout ~ mean_z + slope_z + " + " + ".join(COV), sp).fit(disp=0)
    sp["p_obs"] = 1 - ps.predict(sp)
    pobs = 1 - sp.dropout.mean()
    sp["ipw"] = np.clip(pobs / np.clip(sp.p_obs, 0.05, 1), 0.2, 5.0)
    dw = d.copy()
    dw["ipw"] = dw.mergeid.map(sp.set_index("mergeid").ipw)
    r = fit(FML, dw, weights=dw.ipw)
    b, _, _ = term(r, "S7_dropout", "IPW", "time:slope_z", dw.mergeid.nunique())
    SIGN["S7_IPW"] = b
    inc = (s.adl9 - s.adl8).dropna()
    worst = inc.quantile(.90) if len(inc) else 1.0
    s2 = s.copy()
    fill_from = s2.adl8.fillna(s2.adl7).fillna(s2.adl6)
    s2["adl9"] = s2.adl9.fillna(fill_from + worst)
    d2 = make_long(s2, [7, 8, 9], "adl")
    d2 = d2[d2.y.notna()].copy()
    r = fit(FML, d2)
    b, _, _ = term(r, "S7_dropout", "worst_case(+%.2f)" % worst,
                   "time:slope_z", d2.mergeid.nunique())
    SIGN["S7_worst_case"] = b
except Exception as e:
    w("  S7 ERROR %r" % e)
    w(traceback.format_exc()[-500:])

# S8 binary worsening event
try:
    ev = s.copy()
    ev["worse"] = (ev.adl9 > ev.adl6).astype(float)
    ev.loc[ev.adl9.isna(), "worse"] = np.nan
    ev = prep(ev[ev.worse.notna()].copy())
    lr = smf.logit("worse ~ mean_z + slope_z + " + " + ".join(COV), ev).fit(disp=0)
    bb = lr.params["slope_z"]
    rec("S8_event", "worsening_event", "slope_z_OR", np.exp(bb), lr.bse["slope_z"],
        lr.pvalues["slope_z"], np.exp(lr.conf_int().loc["slope_z"][0]),
        np.exp(lr.conf_int().loc["slope_z"][1]), len(ev),
        "events=%d" % int(ev.worse.sum()))
    SIGN["S8_worsening"] = bb
except Exception as e:
    w("  S8 ERROR %r" % e)

# ---------------------------------------------------------------- 4. negative controls
sec("4. PRE-SPECIFIED BIAS CALIBRATION (negative controls / temporal gradient)")
NC_FAIL = False
try:
    sn = s.copy()
    # NC1: pre-exposure disability
    nc1 = sn[sn.adl_w2.notna()].copy()
    r = smf.ols("adl_w2 ~ slope_z + mean_z + " + " + ".join(COV_NOADL), nc1).fit(cov_type="HC3")
    b, _, p = term(r, "NC1", "pre_exposure_ADL", "slope_z", len(nc1))
    CRIT["NC1"] = bool(p > 0.05)
    NC_FAIL |= not CRIT["NC1"]
    w("    NC1 (should be null): %s" % ("PASS" if CRIT["NC1"] else "FAIL"))
    # NC2: balance on fixed baseline traits
    bal = [c for c in ["age_w2", "female", "educl", "rabyear", "w2_mheight", "cesd_w2"]
           if c in sn.columns]
    for c in bal:
        rr = smf.ols("slope_z ~ %s" % c, sn).fit(cov_type="HC3")
        w("    NC2 balance %-14s beta=%+.4f P=%.4g" %
          (c, rr.params[c], rr.pvalues[c]))
    rr = smf.ols("slope_z ~ " + " + ".join(bal), sn).fit(cov_type="HC3")
    rest = " = 0, ".join(bal) + " = 0"
    wt = rr.wald_test(rest, scalar=True)
    CRIT["NC2_joint_P"] = float(wt.pvalue)
    CRIT["NC2"] = bool(wt.pvalue > 0.05)
    NC_FAIL |= not CRIT["NC2"]
    w("    NC2 joint Wald P=%.4g (should be >0.05): %s" %
      (wt.pvalue, "PASS" if CRIT["NC2"] else "FAIL"))
    # NC3: temporal gradient (per-wave change)
    sn["d_pre"] = (sn.adl6 - sn.adl_w2) / 4.0     # 4 wave-intervals, concurrent
    sn["d_post"] = (sn.adl9 - sn.adl7) / 2.0      # 2 wave-intervals, prospective
    nc3_p = {}
    for lab, col in [("concurrent_w2-w6", "d_pre"), ("prospective_w7-w9", "d_post")]:
        sub = sn[sn[col].notna()].copy()
        rr = smf.ols("%s ~ slope_z + mean_z + %s" % (col, " + ".join(COV_NOADL)),
                     sub).fit(cov_type="HC3")
        b, _, p = term(rr, "NC3", lab, "slope_z", len(sub))
        CRIT["NC3_" + col] = b
        nc3_p[col] = p
    pre, post = CRIT.get("NC3_d_pre", np.nan), CRIT.get("NC3_d_post", np.nan)
    CRIT["NC3"] = bool(np.isfinite(pre) and np.isfinite(post) and pre < post)
    NC_FAIL |= not CRIT["NC3"]
    w("    NC3 temporal gradient (concurrent %.4f vs prospective %.4f per wave): %s" %
      (pre, post, "PASS" if CRIT["NC3"] else "FAIL"))
    w("    NC3 CAVEAT (frozen plan limitation): the criterion compares COEFFICIENTS only.")
    w("    Here the concurrent term is P=%.4g while the prospective term is P=%.4g;"
      " a concurrent association is therefore more precisely detected than a prospective one." %
      (nc3_p.get("d_pre", np.nan), nc3_p.get("d_post", np.nan)))
    w("    This must be reported as a limitation of the pre-set criterion, not re-judged ad hoc.")
except Exception as e:
    w("  NC ERROR %r" % e)
    w(traceback.format_exc()[-600:])
CRIT["C4"] = not NC_FAIL

# ---------------------------------------------------------------- 5. resampling
sec("5. S9: CLUSTER BOOTSTRAP / PERMUTATION / SPLIT-HALF")
boot, perm = [], []
try:
    Xb = dmatrix(FML.split("~", 1)[1], d, return_type="dataframe")
    cols = list(Xb.columns)
    Xv = Xb.to_numpy(float)
    yv = d.y.to_numpy(float)
    tv = d.time.to_numpy(float)
    idx_slope = cols.index("time:slope_z")
    i_s, i_m = cols.index("slope_z"), cols.index("mean_z")
    i_ts, i_tm = cols.index("time:slope_z"), cols.index("time:mean_z")
    b_ols = float(np.linalg.lstsq(Xv, yv, rcond=None)[0][idx_slope])
    w("  OLS point estimate (for resampling calibration) = %.6f ; cluster-robust = %.6f"
      % (b_ols, b0))
    pid_arr = d.mergeid.to_numpy()
    pids = np.unique(pid_arr)
    row_by_pid = {p: np.where(pid_arr == p)[0] for p in pids}
    pv = s.set_index("mergeid")[["mean_z", "slope_z"]]
    B = 1000
    for _ in range(B):
        draw = rng.choice(pids, size=len(pids), replace=True)
        rows_i = np.concatenate([row_by_pid[p] for p in draw])
        boot.append(float(np.linalg.lstsq(Xv[rows_i], yv[rows_i], rcond=None)[0][idx_slope]))
    boot = np.array(boot)
    pboot = float((boot > 0).mean())
    w("  cluster bootstrap B=%d : median=%+.4f  mean=%+.4f  2.5/97.5%%=[%+.4f,%+.4f]  P(beta>0)=%.4f"
      % (B, np.median(boot), boot.mean(), np.quantile(boot, .025),
         np.quantile(boot, .975), pboot))
    CRIT["C3_boot_P"] = pboot
    for _ in range(B):
        pp = rng.permutation(pids)
        mp = dict(zip(pids, pp))
        src = np.array([mp[p] for p in pid_arr])
        sp = pv.loc[src, "slope_z"].to_numpy()
        mz = pv.loc[src, "mean_z"].to_numpy()
        Xp = Xv.copy()
        Xp[:, i_s] = sp
        Xp[:, i_ts] = tv * sp
        Xp[:, i_m] = mz
        Xp[:, i_tm] = tv * mz
        perm.append(float(np.linalg.lstsq(Xp, yv, rcond=None)[0][idx_slope]))
    perm = np.array(perm)
    pperm = (1 + np.sum(np.abs(perm) >= abs(b_ols))) / (B + 1)
    w("  permutation B=%d : SD=%.4f  2.5/97.5%%=[%+.4f,%+.4f]  two-sided P=%.4f"
      % (B, perm.std(), np.quantile(perm, .025), np.quantile(perm, .975), pperm))
    CRIT["C3_perm_P"] = pperm
    pd.DataFrame({"beta": boot}).to_csv(os.path.join(OUT, "prespec_bootstrap.csv"), index=False)
    pd.DataFrame({"beta": perm}).to_csv(os.path.join(OUT, "prespec_perm.csv"), index=False)
    # split half
    order = rng.permutation(pids)
    half = len(order) // 2
    for lab, pp in [("half_A", order[:half]), ("half_B", order[half:])]:
        sub = d[d.mergeid.isin(set(pp))]
        rr = fit(FML, sub)
        term(rr, "S9_split", lab, "time:slope_z", sub.mergeid.nunique())
except Exception as e:
    w("  S9 ERROR %r" % e)
    w(traceback.format_exc()[-600:])
CRIT["C3"] = bool(CRIT.get("C3_boot_P", 0) >= 0.975 and CRIT.get("C3_perm_P", 1) < 0.05)

# ---------------------------------------------------------------- 6. power
sec("6. S10: POWER / MINIMUM DETECTABLE EFFECT / CLINICAL SCALE")
zcrit, zpow = 1.959964, 0.841621
power = stats.norm.cdf(abs(b0 / se0) - zcrit) + stats.norm.cdf(-abs(b0 / se0) - zcrit)
mde = (zcrit + zpow) * se0
se_target = abs(b0) / (zcrit + zpow)
n_need = d.mergeid.nunique() * (se0 / se_target) ** 2
w("  observed beta=%.4f SE=%.4f  z=%.3f  P=%.4g" % (b0, se0, b0 / se0, p0))
w("  observed power (two-sided alpha=.05, for the OBSERVED effect) = %.1f%%" % (100 * power))
w("  MDE at 80%% power = %.4f ADL points / wave / 1 SD slope" % mde)
w("  over w7->w9 (2 waves) MDE = %.4f ADL points ; observed 1-SD effect = %.4f ADL points"
  % (2 * mde, 2 * b0))
w("  approximate n needed for 80%% power at this effect = %.0f persons (vs %d available, x%.2f)"
  % (n_need, d.mergeid.nunique(), n_need / d.mergeid.nunique()))
CRIT["C7_power"] = float(power)
CRIT["C7_MDE"] = float(mde)

# ---------------------------------------------------------------- 7. CHARLS
sec("7. S11: CHARLS REPLICATION, SAME FROZEN SPECIFICATION")
try:
    ch = pd.read_csv(os.path.join(OUT, "charls_base.csv"), dtype={"pid": str})
    ch["pid"] = ch.pid.astype(str).str.strip()
    w1 = ch[ch.wave == 1].set_index("pid")
    covcols = ["age2011", "ragender", "raeducl", "bmi", "smoken", "hibpe", "diabe", "hearte"]
    for c in covcols:
        if c not in w1.columns:
            w1[c] = np.nan
        w1[c] = pd.to_numeric(w1[c], errors="coerce")
        w1[c] = w1[c].fillna(w1[c].median() if w1[c].notna().any() else 0)
    w1["adl1"] = pd.to_numeric(w1.adlab_c, errors="coerce")
    w1["adl1"] = w1.adl1.fillna(w1.adl1.median())
    exp = ch[ch.wave == 1][["pid", "dcrp"]].dropna().set_index("pid")
    common = w1.index.intersection(exp.index)
    lr_rows = []
    for wv in [4, 5]:
        dw = ch[ch.wave == wv].set_index("pid")
        tmp = pd.DataFrame(index=common)
        tmp["pid"] = common
        tmp["y"] = pd.to_numeric(dw.adlab_c.reindex(common), errors="coerce").values
        tmp["time"] = wv - 4
        lr_rows.append(tmp)
    dd = pd.concat(lr_rows, ignore_index=True)
    dd = dd[dd.y.notna()].copy()
    dd["dcrp_z"] = z(dd.pid.map(exp.dcrp))
    for c in covcols + ["adl1"]:
        dd[c] = dd.pid.map(w1[c])
    f = "y ~ time * dcrp_z + " + " + ".join(covcols + ["adl1"])
    rr = fit(f, dd, key="pid")
    bc, _, pc = term(rr, "S11_CHARLS", "w4-w5_prospective", "time:dcrp_z",
                     dd.pid.nunique())
    CRIT["C6"] = bool(bc > 0)
    w("    C6 (CHARLS direction beta>0): %s" % ("PASS" if CRIT["C6"] else "FAIL"))
    # CHARLS negative control: balance at 2011 baseline
    bal = pd.DataFrame(index=common)
    bal["pid"] = common
    bal["dcrp_z"] = z(bal.pid.map(exp.dcrp)).to_numpy(dtype=float)
    for c in ["age2011", "ragender", "raeducl", "adl1"]:
        bal[c] = bal.pid.map(w1[c]).to_numpy()
    rn = smf.ols("dcrp_z ~ age2011 + ragender + raeducl + adl1", bal).fit(cov_type="HC3")
    wt = rn.wald_test("age2011 = 0, ragender = 0, raeducl = 0, adl1 = 0", scalar=True)
    w("    CHARLS NC2 balance joint Wald P=%.4g (should be >0.05)" % wt.pvalue)
    rec("S11_CHARLS", "balance_NC", "joint_Wald_P", np.nan, np.nan, float(wt.pvalue),
        np.nan, np.nan, len(bal))
except Exception as e:
    w("  S11 ERROR %r" % e)
    w(traceback.format_exc()[-600:])

# ---------------------------------------------------------------- 8. multiplicity
sec("8. MULTIPLICITY OVER ALL SECONDARY/SENSITIVITY TESTS")
sec_rows = [r for r in ROWS if r["section"] not in ("PRIMARY",) and
            np.isfinite(r["p"]) and r["term"] not in ("time:mean_z",)]
if sec_rows:
    pv = np.array([r["p"] for r in sec_rows])
    o = np.argsort(pv)
    q = np.empty_like(pv)
    q[o] = np.minimum.accumulate((pv[o] * len(pv) / (np.arange(len(pv)) + 1))[::-1])[::-1]
    q = np.clip(q, 0, 1)
    for r, qq in zip(sec_rows, q):
        r["q_BH"] = qq
    w("  secondary/sensitivity tests = %d ; raw P<0.05 = %d ; BH q<0.05 = %d ; min q=%.4g"
      % (len(pv), (pv < .05).sum(), (q < .05).sum(), q.min()))
    for r in sorted([x for x in sec_rows if x["p"] < .05], key=lambda x: x["p"]):
        w("    raw P<0.05 : %-30s %-24s P=%.4g q=%.4g" %
          (r["label"], r["term"], r["p"], r.get("q_BH", np.nan)))

# ---------------------------------------------------------------- 9. criteria
sec("9. FROZEN SUCCESS CRITERIA: ITEM-BY-ITEM VERDICT")
CRIT["C2"] = all(v > 0 for k, v in SIGN.items() if np.isfinite(v))
w("  C1 primary (P<0.05, beta>0)                 : %s  [beta=%+.4f P=%.4g]"
  % ("PASS" if CRIT["C1"] else "FAIL", b0, p0))
w("  C2 all sensitivity signs positive          : %s  [%d/%d]" %
  ("PASS" if CRIT["C2"] else "FAIL",
   sum(1 for v in SIGN.values() if np.isfinite(v) and v > 0), len(SIGN)))
for k, v in SIGN.items():
    w("       %-28s beta=%+.4f  %s" % (k, v, "+" if v > 0 else "-"))
w("  C3 bootstrap P(beta>0)>=.975 AND perm P<.05: %s  [boot=%.4f perm=%.4f]" %
  ("PASS" if CRIT["C3"] else "FAIL", CRIT.get("C3_boot_P", np.nan),
   CRIT.get("C3_perm_P", np.nan)))
if CRIT.get("C3_boot_P", 0) == 0.975:
    w("       C3 CAVEAT: the bootstrap value sits EXACTLY on the frozen 0.975 threshold"
      " (975/1000 draws); with a strict '>' rule or a different B it could flip."
      " Do not treat this as stable evidence.")
w("  C4 all negative controls null              : %s  [NC1=%s NC2=%s NC3=%s]" %
  ("PASS" if CRIT["C4"] else "FAIL", CRIT.get("NC1"), CRIT.get("NC2"), CRIT.get("NC3")))
w("  C5 orthogonal slope beta>0 and P<0.10      : %s  [beta=%+.4f P=%.4g]" %
  ("PASS" if CRIT["C5"] else "FAIL", bo, po))
w("  C6 CHARLS direction beta>0                 : %s" % ("PASS" if CRIT.get("C6") else "FAIL"))
w("  C7 observed power=%.1f%%  MDE=%.4f/wave" % (100 * CRIT["C7_power"], CRIT["C7_MDE"]))
keys = ["C1", "C2", "C3", "C4", "C5", "C6"]
npass = sum(1 for k in keys if CRIT.get(k))
w("\n  PASSED %d / %d criteria" % (npass, len(keys)))
# Rule order follows SAP section 7 verbatim: the level-coupling / bias rule
# takes precedence over the generic "suggestive" rule when C5 or C4 fails.
if npass == len(keys):
    verdict = ("ALL PASS -- internally consistent, STILL hypothesis-generating "
               "(SAP section 0)")
elif CRIT.get("C4") is False or CRIT.get("C5") is False:
    verdict = ("NOT CONFIRMED -- S3 shows the signal is not separable from baseline "
               "inflammation level (mathematical coupling / regression to the mean); "
               "abandon as a manuscript main line [SAP sec.7 rule 3]")
elif CRIT["C1"] is False and npass >= 4:
    verdict = ("NOT CONFIRMED -- suggestive but primary test fails; "
               "exploratory secondary only")
else:
    verdict = "NOT CONFIRMED -- insufficient support for a main conclusion"
w("  VERDICT: %s" % verdict)
w("  Reproducibility gate: %s" % ("PASS" if CRIT["REPRODUCIBILITY"] else "FAIL"))

df = pd.DataFrame(ROWS)
df.to_csv(os.path.join(OUT, "prespec_coefs.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame([dict(criterion=k, value=str(v)) for k, v in CRIT.items()]).to_csv(
    os.path.join(OUT, "prespec_criteria.csv"), index=False, encoding="utf-8-sig")
w("\nprespec_coefs.csv rows=%d ; finished %s" %
  (len(df), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
LOG.close()
print("done")
