# -*- coding: utf-8 -*-
"""Review round-2 item 3: treat death / unknown vital status as a WORST outcome.

READ-ONLY with respect to every previously frozen result.  This script does
not rewrite any existing output and does not change the primary analysis.
It writes out/DEATH_SENS.txt and out/death_sens_coefs.csv.

Why a real competing-risk (Fine-Gray) model is NOT possible here is stated
explicitly in the log: the harmonized extract has no post-2012 death events
or death times (iwstat=5 identically 0 at w7-w9; radyear max 2012; EOL years
2002-2012 with zero overlap with Design B).  We therefore implement
   (a) death-as-worst-ADL bound analyses (scale ceiling),
   (b) a tipping-point grid,
   (c) a composite endpoint and a discrete-time hazard of 'unknown vital
       status' onset as approximations,
and label all of them as bound/approximation analyses, NOT as a competing-risk
model with observed deaths.
"""
import os
import traceback
import numpy as np
import pandas as pd
import psycopg2
import statsmodels.formula.api as smf


def _db(name):
    """Open a local PostgreSQL connection from environment variables."""
    import psycopg2
    return psycopg2.connect(
        host=os.environ.get("ELSA_DB_HOST", "localhost"),
        port=int(os.environ.get("ELSA_DB_PORT", "5432")),
        user=os.environ.get("ELSA_DB_USER", "postgres"),
        password=os.environ.get("ELSA_DB_PASSWORD", ""),
        dbname=name,
    )


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "DEATH_SENS.txt"), "w", encoding="utf-8")


def w(*a):
    print(*a, file=LOG)
    LOG.flush()


def sec(t):
    w("\n" + "=" * 92)
    w("## " + t)
    w("=" * 92)


COV = ["age_w2", "female", "educl", "adl6", "bmi_w2", "smoken_w2", "n_comorb"]
FML = "y ~ time * mean_z + time * slope_z + " + " + ".join(COV)
rows = []


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


# --------------------------------------------------------------- build cohort
ids = set(pd.read_csv(os.path.join(OUT, "cohort_designB.csv"),
                      dtype={"mergeid": str}).mergeid.astype(str).str.strip())
base = pd.read_csv(os.path.join(OUT, "elsa_base.csv"), dtype={"mergeid": str})
base["mergeid"] = base.mergeid.astype(str).str.strip()
long = pd.read_csv(os.path.join(OUT, "exposure_long.csv"), dtype={"mergeid": str})
long["mergeid"] = long.mergeid.astype(str).str.strip()

s = prep(base[base.mergeid.isin(ids)].copy())
ex = indiv(long[long.mergeid.isin(ids)], "logcrp", [0, 1, 2])
s = s.merge(ex, on="mergeid", how="inner")
for c, src in [("mean_z", "mean_y"), ("slope_z", "slope_y"), ("base_z", "base_y")]:
    s[c] = z(s[src])

# --------------------------------------------------------------- iwstat
CE = _db("elsa")
iwq = pd.read_sql(
    "SELECT p1.mergeid, p1.r7iwstat AS iw7, p1.r8iwstat AS iw8, p1.r9iwstat AS iw9, "
    "k.radyear FROM gh_elsa_p1 p1 JOIN elsa_keys k USING (mergeid)", CE)
iwq["mergeid"] = iwq.mergeid.astype(str).str.strip()
for c in ["iw7", "iw8", "iw9", "radyear"]:
    iwq[c] = pd.to_numeric(iwq[c], errors="coerce")
_iw = iwq.set_index("mergeid")
for c in ["iw7", "iw8", "iw9", "radyear"]:
    s[c] = s.mergeid.map(_iw[c])

sec("0. DATA AND WHY A REAL COMPETING-RISK MODEL IS NOT POSSIBLE")
w("Design B n=%d ; w2/w4/w6 log-CRP complete." % len(s))
w("Observed ADL range: min=%.3f max=%.3f ; 99th pct=%.3f" %
  (np.nanmin(s[["adl7", "adl8", "adl9"]].values),
   np.nanmax(s[["adl7", "adl8", "adl9"]].values),
   np.nanpercentile(s[["adl7", "adl8", "adl9"]].values, 99)))
for wv in [7, 8, 9]:
    w("  w%d : iwstat=1 (interviewed) %3d ; iwstat=9 (status unknown) %3d ; other/NaN %3d ; "
      "ADL non-null %3d ; missing %3d" %
      (wv, int((s["iw%d" % wv] == 1).sum()), int((s["iw%d" % wv] == 9).sum()),
       int((~s["iw%d" % wv].isin([1, 9])).sum()),
       int(s["adl%d" % wv].notna().sum()), int(s["adl%d" % wv].isna().sum())))
w("  radyear non-null in B: %d ; max radyear in full sample: 2012 (see REVIEW_AUDIT.txt)" %
  int(s.radyear.notna().sum()))
w("")
w("A Fine-Gray / cause-specific competing-risk model requires an observed death")
w("indicator AND a death/censoring time.  Here iwstat=5 (died since last wave) is")
w("identically 0 at w7-w9, iwstat=6 is constant, radyear stops at 2012, and the")
w("end-of-life module has zero overlap with Design B.  Hence deaths cannot be")
w("identified or timed.  The analyses below are BOUND / APPROXIMATION analyses:")
w("persons of unknown vital status are assigned the worst possible disability")
w("value, or counted as a composite/hazard event.  They bound how much the")
w("unknown-death mixture could change the trajectory coefficient; they are NOT")
w("a competing-risk model with observed mortality.")


def make_long(s, waves, ycol="adl"):
    rec = []
    for wv in waves:
        d = s[["mergeid"]].copy()
        d["y"] = pd.to_numeric(s["%s%d" % (ycol, wv)], errors="coerce")
        for c in COV:
            d[c] = s[c]
        d["time"] = wv - waves[0]
        d["wave"] = wv
        d["iw"] = s["iw%d" % wv]
        rec.append(d)
    return pd.concat(rec, ignore_index=True)


def fit(formula, d):
    return smf.ols(formula, d).fit(cov_type="cluster",
                                   cov_kwds={"groups": d["mergeid"]})


def report(label, r, n_person, n_obs, note=""):
    t = "time:slope_z"
    ci = r.conf_int().loc[t].tolist()
    rec = dict(scenario=label, beta=r.params[t], se=r.bse[t], p=r.pvalues[t],
               ci_lo=ci[0], ci_hi=ci[1], n_person=n_person, n_obs=n_obs, note=note)
    rows.append(rec)
    w("  %-42s beta=%+.4f  SE=%.4f  95%%CI [%+.4f,%+.4f]  P=%.4g  n=%d/%d" %
      (label, rec["beta"], rec["se"], rec["ci_lo"], rec["ci_hi"], rec["p"],
       n_person, n_obs))
    return rec


WAVES = [7, 8, 9]
d0 = make_long(s, WAVES)
for c in ["mean_z", "slope_z", "base_z"]:
    d0[c] = d0.mergeid.map(s.set_index("mergeid")[c])

sec("1. REFERENCE: complete-case (delete missing outcome) -- reproduces main model")
dcc = d0[d0.y.notna()].copy()
r0 = fit(FML, dcc)
report("S0 complete-case (primary specification)", r0,
       dcc.mergeid.nunique(), len(dcc), "reference; should match beta=+0.0568")
w("    time:mean_z beta=%+.4f P=%.4g" %
  (r0.params.get("time:mean_z", np.nan), r0.pvalues.get("time:mean_z", np.nan)))

sec("2. EXISTING worst-case (LOCF + 90th pct observed increment) -- reproduce")
obs_inc = (s.adl9 - s.adl8).dropna()
worst = obs_inc.quantile(0.90) if len(obs_inc) else 1.0
_fill = s.adl8.fillna(s.adl7).fillna(s.adl6)
w("  90th pct of observed w8->w9 ADL increment = %.4f" % worst)
s1 = s.copy()
s1["adl9"] = s1.adl9.fillna(_fill + worst)
d1 = make_long(s1, WAVES)
for c in ["mean_z", "slope_z", "base_z"]:
    d1[c] = d1.mergeid.map(s1.set_index("mergeid")[c])
d1 = d1[d1.y.notna()]
report("S1 LOCF + p90 increment (existing)", fit(FML, d1),
       d1.mergeid.nunique(), len(d1), "existing worst-case imputation")

sec("3. DEATH-AS-WORST-OUTCOME (new, reviewer item 3)")
ceiling = float(np.nanmax(s[["adl7", "adl8", "adl9"]].values))
p99 = float(np.nanpercentile(s[["adl7", "adl8", "adl9"]].values, 99))
w("  ADL scale ceiling used as 'worst' = %.3f ; observed 99th pct = %.3f" %
  (ceiling, p99))


def fill_worst(mode):
    """Return a copy of d0 with missing y filled; only rows with unknown vital
    status get the worst value.  mode: 'ceiling' | 'p99' | 'persistent'."""
    d = d0.copy()
    locf = d.groupby("mergeid")["y"].ffill()
    # for wave 7 there is no earlier outcome row; use w6 baseline ADL
    miss = d.y.isna()
    first = d.groupby("mergeid").cumcount() == 0
    locf = locf.where(~first, d.mergeid.map(s.set_index("mergeid").adl6))
    unknown = d.iw.ne(1)          # NaN also counts as unknown
    if mode == "persistent":
        # once a person has unknown status, all later missing waves are worst
        d["_ever_unknown"] = unknown.groupby(d.mergeid).cummax()
        flag = miss & d._ever_unknown
    else:
        flag = miss & unknown
    val = ceiling if mode in ("ceiling", "persistent") else p99
    d.loc[flag, "y"] = val
    # any remaining missing (interviewed but no ADL) -> LOCF
    d["y"] = d.y.fillna(locf)
    return d, int(flag.sum())


for mode, lab in [("ceiling", "S2 death-as-worst ADL=ceiling, per wave"),
                  ("persistent", "S2b death-as-worst, persist after unknown"),
                  ("p99", "S3 death-as-worst ADL=99th pct")]:
    dd, nf = fill_worst(mode)
    dd = dd[dd.y.notna()].copy()
    report(lab, fit(FML, dd), dd.mergeid.nunique(), len(dd),
           "worst-value rows filled=%d" % nf)

sec("4. TIPPING-POINT GRID (delta added to LOCF for unknown-vital-status rows)")
_fill_long = d0.groupby("mergeid")["y"].ffill()
_first = d0.groupby("mergeid").cumcount() == 0
_fill_long = _fill_long.where(~_first,
                              d0.mergeid.map(s.set_index("mergeid").adl6))
_unknown = d0.iw.ne(1)
_miss = d0.y.isna() & _unknown
for delta in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    dd = d0.copy()
    dd.loc[_miss, "y"] = _fill_long[_miss] + delta
    dd["y"] = dd.y.fillna(_fill_long)
    dd = dd[dd.y.notna()].copy()
    report("S4 tipping delta=+%.1f" % delta, fit(FML, dd),
           dd.mergeid.nunique(), len(dd), "unknown-status rows=%d" % int(_miss.sum()))

sec("5. COMPOSITE ENDPOINT + DISCRETE-TIME HAZARD (approximations)")
# Composite: unknown vital status at w9 OR observed ADL worsening w6->w9
s["comp"] = np.where(s.adl9.isna() & s.iw9.ne(1), 1.0,
                     np.where(s.adl9 > s.adl6, 1.0, 0.0))
s.loc[s.adl9.isna() & s.iw9.eq(1), "comp"] = np.nan
ev = s[s.comp.notna()].copy()
try:
    lr = smf.logit("comp ~ slope_z + mean_z + " + " + ".join(COV), ev).fit(disp=0)
    b, p = lr.params["slope_z"], lr.pvalues["slope_z"]
    w("  composite event = %d / %d (%.1f%%): OR(slope_z)=%.4f  P=%.4g" %
      (int(ev.comp.sum()), len(ev), 100 * ev.comp.mean(), np.exp(b), p))
    rows.append(dict(scenario="S5 composite endpoint (logistic OR)", beta=np.exp(b),
                     se=np.nan, p=p, ci_lo=np.exp(lr.conf_int().loc["slope_z"][0]),
                     ci_hi=np.exp(lr.conf_int().loc["slope_z"][1]),
                     n_person=len(ev), n_obs=len(ev),
                     note="event = w9 unknown vital status OR ADL9>ADL6"))
except Exception as e:
    w("  composite ERROR %r" % e)

# Discrete-time hazard: first wave at which vital status becomes unknown.
hz = []
for wv in WAVES:
    d = s[["mergeid", "slope_z", "mean_z"] + COV].copy()
    prev_unknown = s["iw%d" % (wv - 1)].ne(1) if wv > 7 else pd.Series(False, index=s.index)
    cur_unknown = s["iw%d" % wv].ne(1)
    d["event"] = (cur_unknown & ~prev_unknown).astype(float)
    d["time"] = wv - 7
    d["at_risk"] = (~prev_unknown).astype(float)
    hz.append(d[d.at_risk == 1])
hd = pd.concat(hz, ignore_index=True)
try:
    hr = smf.logit("event ~ slope_z + mean_z + time + " + " + ".join(COV),
                   hd).fit(disp=0)
    b, p = hr.params["slope_z"], hr.pvalues["slope_z"]
    w("  discrete-time hazard of 'unknown vital status' onset: events=%d / %d person-waves; "
      "OR(slope_z)=%.4f P=%.4g" % (int(hd.event.sum()), len(hd), np.exp(b), p))
    rows.append(dict(scenario="S6 discrete-time hazard (logistic OR)", beta=np.exp(b),
                     se=np.nan, p=p, ci_lo=np.exp(hr.conf_int().loc["slope_z"][0]),
                     ci_hi=np.exp(hr.conf_int().loc["slope_z"][1]),
                     n_person=hd.mergeid.nunique(), n_obs=len(hd),
                     note="event = first wave with iwstat!=1; NOT observed death"))
except Exception as e:
    w("  hazard ERROR %r" % e)
    w(traceback.format_exc()[-400:])

sec("6. SUMMARY")
w("  Primary (complete-case) beta = %+.4f P=%.4g" %
  (rows[0]["beta"], rows[0]["p"]))
w("  All bound/approximation scenarios:")
for r in rows[1:]:
    w("    %-46s beta=%+.4f P=%.4g" % (r["scenario"], r["beta"], r["p"]))
w("")
w("  Interpretation: these are UPPER-BOUND stress tests.  If the trajectory")
w("  coefficient is not robust to assigning the worst possible outcome to")
w("  persons of unknown vital status, the conclusion must be stated as")
w("  conditional on unverifiable post-2012 vital status, and the data")
w("  limitation must appear in the abstract (not only in Methods 2.3).")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "death_sens_coefs.csv"),
                          index=False, encoding="utf-8-sig")
w("\nDONE")
LOG.close()
print("done")
