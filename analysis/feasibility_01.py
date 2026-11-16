# -*- coding: utf-8 -*-
"""Direction 3 feasibility audit.
   ELSA: arthritis x repeated CRP/fibrinogen -> ADL/IADL trajectory + death.
   Produces out/feasibility.txt (human readable), out/crp_long.csv,
   out/cohort_ids.csv, out/charls_crp.csv.
"""
import os
import psycopg2
import pandas as pd
import numpy as np


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
RDATA = os.environ.get("ELSA_RDATA",
                        os.path.join(BASE, "data", "AGING_DB_R"))
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "feasibility.txt"), "w", encoding="utf-8")


def w(*a):
    print(*a, file=LOG)
    LOG.flush()


def sec(t):
    w("\n" + "=" * 92)
    w("## " + t)
    w("=" * 92)


CR = _db("elsa_raw")
CE = _db("elsa")
CC = _db("charls")
for c in (CR, CE, CC):
    cur = c.cursor()
    cur.execute("SET statement_timeout='300s'")
    cur.close()

# ------------------------------------------------------------------
sec("1. Is w8nurse a separate wave from w89nurse?")
w("A) visit-year distribution")
for t in ["rw_w2nurse_p1", "rw_w4nurse_p1", "rw_w6nurse_p1",
          "rw_w8nurse_p1", "rw_w89nurse_p1"]:
    d = pd.read_sql(
        "SELECT visyear, count(*) AS n_row, count(DISTINCT mergeid) AS n_person "
        "FROM %s GROUP BY visyear ORDER BY visyear" % t, CR)
    w("\n%s:" % t)
    w(d.to_string(index=False))

w("\nB) duplicate mergeid in w89nurse: %s" % pd.read_sql(
    "SELECT count(*) AS n_dup_id FROM (SELECT mergeid FROM rw_w89nurse_p1 "
    "GROUP BY mergeid HAVING count(*)>1) x", CR).to_string(index=False))

w("\nC) w8nurse records whose CRP value is IDENTICAL to the same person in w89nurse")
w(pd.read_sql("""
SELECT count(*) AS n_join,
       count(*) FILTER (WHERE a.hscrp = b.hscrp) AS n_same_crp,
       count(*) FILTER (WHERE a.cfib  = b.cfib)  AS n_same_fib,
       count(*) FILTER (WHERE a.hscrp IS DISTINCT FROM b.hscrp) AS n_diff_crp
FROM rw_w8nurse_p1 a JOIN rw_w89nurse_p1 b USING (mergeid)
""", CR).to_string(index=False))
w(">>> If n_same_crp == n_join, w89nurse is a SUPERSET of w8nurse (one measurement),")
w("    i.e. only FOUR distinct time points, not five.")

# ------------------------------------------------------------------
sec("2. Build cleaned CRP / fibrinogen long table (4 time points)")
WAVE_MAP = [("rw_w2nurse_p1", 2, 2004), ("rw_w4nurse_p1", 4, 2008),
            ("rw_w6nurse_p1", 6, 2012), ("rw_w89nurse_p1", 89, 2016)]
frames = []
for t, wv, yr in WAVE_MAP:
    d = pd.read_sql("SELECT mergeid, hscrp AS crp, cfib AS fib, visyear FROM %s" % t, CR)
    d["wave_raw"] = wv
    d["year"] = yr
    frames.append(d)
    w("%-18s rows=%5d  persons=%5d" % (t, len(d), d.mergeid.nunique()))
raw = pd.concat(frames, ignore_index=True)
w("\nlong table (raw, duplicate IDs kept): %d rows, %d persons" %
  (len(raw), raw.mergeid.nunique()))

# person-level dedup (w89nurse has 2 duplicated ids)
raw = raw.sort_values(["mergeid", "wave_raw", "crp"]).drop_duplicates(
    subset=["mergeid", "wave_raw"], keep="first")
w("after (person,wave) dedup: %d rows, %d persons" %
  (len(raw), raw.mergeid.nunique()))

# ---- cleaning audit ----
w("\n[Cleaning step audit]")
w("total person-wave rows: %d" % len(raw))
w("  CRP NULL                     : %d" % raw.crp.isna().sum())
w("  CRP negative sentinel (<0)  : %d   <- ALL negative codes, not just -1/-11" %
  (raw.crp < 0).sum())
w("     breakdown: %s" % raw.loc[raw.crp < 0, "crp"].value_counts().sort_index().to_dict())
w("  CRP == 0                    : %d" % (raw.crp == 0).sum())
w("  CRP > 10 (acute infection)  : %d" % (raw.crp > 10).sum())
w("  CRP > 100 (extreme)         : %d" % (raw.crp > 100).sum())
w("  FIB negative sentinel (<0)  : %d" % (raw.fib < 0).sum())
w("  FIB == 0                    : %d" % (raw.fib == 0).sum())
w("  FIB > 10 (implausible)      : %d" % (raw.fib > 10).sum())

raw["crp_valid"] = (raw.crp > 0) & (raw.crp <= 10)
raw["fib_valid"] = (raw.fib > 0) & (raw.fib <= 10)
raw["crp_valid_no10"] = raw.crp > 0          # sensitivity: keep CRP>10
w("\n  FINAL valid CRP (0<crp<=10): %d rows / %d persons" %
  (raw.crp_valid.sum(), raw.loc[raw.crp_valid, "mergeid"].nunique()))
w("  FINAL valid FIB (0<fib<=10): %d rows / %d persons" %
  (raw.fib_valid.sum(), raw.loc[raw.fib_valid, "mergeid"].nunique()))
w("  both CRP+FIB valid          : %d rows / %d persons" %
  ((raw.crp_valid & raw.fib_valid).sum(),
   raw.loc[raw.crp_valid & raw.fib_valid, "mergeid"].nunique()))

w("\n[Per wave: valid CRP / FIB after full cleaning]")
w(raw.groupby(["wave_raw", "year"]).agg(
    n_row=("mergeid", "size"),
    n_crp_valid=("crp_valid", "sum"),
    n_fib_valid=("fib_valid", "sum"),
    crp_mean=("crp", lambda s: round(s[(s > 0) & (s <= 10)].mean(), 3)),
    crp_median=("crp", lambda s: round(s[(s > 0) & (s <= 10)].median(), 3)),
    fib_mean=("fib", lambda s: round(s[(s > 0) & (s <= 10)].mean(), 3)),
).to_string())

# number of valid waves per person
cnt = raw[raw.crp_valid].groupby("mergeid").size().rename("n_crp_waves")
cnt_no10 = raw[raw.crp_valid_no10].groupby("mergeid").size().rename("n_crp_waves_no10")
w("\n[Distribution of number of valid CRP waves per person]")
w(cnt.value_counts().sort_index().to_string())
w("\n  >=2 waves: %d   >=3 waves: %d   =4 waves: %d" %
  ((cnt >= 2).sum(), (cnt >= 3).sum(), (cnt == 4).sum()))
w("  sensitivity (keep CRP>10): >=3 waves: %d  =4 waves: %d" %
  ((cnt_no10 >= 3).sum(), (cnt_no10 == 4).sum()))
w("\n[Wave-combination pattern among persons with >=3 valid CRP waves]")
pat = (raw[raw.crp_valid].groupby("mergeid")["wave_raw"]
       .apply(lambda s: ",".join(map(str, sorted(s)))))
w(pat[pat.str.count(",") >= 2].value_counts().to_string())

raw.to_csv(os.path.join(OUT, "crp_long.csv"), index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------
sec("3. Bridge IDs and load harmonized ELSA panel")
keys = pd.read_sql("SELECT mergeid, idauniq FROM elsa_keys", CE)
w("elsa_keys: %d rows; mergeid dtype=%s, idauniq dtype=%s" %
  (len(keys), keys.mergeid.dtype, keys.idauniq.dtype))
w("sample: %s" % keys.head(3).to_dict("records"))

import pyreadr
elsa = pyreadr.read_r(os.path.join(RDATA, "elsa.RData"))["elsa"]
w("\nelsa.RData: %d rows x %d cols" % elsa.shape)
idcols = [c for c in elsa.columns if c.lower() in
          ("mergeid", "idauniq", "id", "pid", "hhidpn")]
w("ID columns present: %s" % idcols)
w("arthritis cols: %s" % [c for c in elsa.columns if "arth" in c.lower()])
w("ADL/IADL cols: %s" % [c for c in elsa.columns if "adl" in c.lower()][:20])
w("iwstat cols: %s" % [c for c in elsa.columns if "iwstat" in c.lower()])
w("wave values: %s" % sorted(elsa["wave"].dropna().unique().tolist()))

# ------------------------------------------------------------------
sec("4. CORE INTERSECTION: arthritis x >=3 valid CRP waves")
# harmonized panel uses idauniq; nurse tables use mergeid
crp_pid = cnt[cnt >= 3].index.to_frame(index=False).rename(columns={"mergeid": "mergeid"})
crp_pid = crp_pid.merge(keys, on="mergeid", how="left")
w("persons with >=3 valid CRP: %d ; matched to idauniq: %d" %
  (len(crp_pid), crp_pid.idauniq.notna().sum()))

elsa["arthre"] = pd.to_numeric(elsa["arthre"], errors="coerce")
w("\narthre value counts (all waves): %s" %
  elsa["arthre"].value_counts(dropna=False).head(10).to_dict())

# arthritis definition A: diagnosed at w2 (index wave)
arth_w2 = set(elsa.loc[(elsa.wave == 2) & (elsa.arthre == 1), "idauniq"].dropna())
# definition B: ever diagnosed at any wave 2..6 (accumulated, exposure window)
arth_cum = set(elsa.loc[(elsa.wave.between(2, 6)) & (elsa.arthre == 1), "idauniq"].dropna())
# definition C: w2 non-missing (denominator)
w2_valid = set(elsa.loc[(elsa.wave == 2) & elsa.arthre.notna(), "idauniq"].dropna())
w("arthritis @w2 (arthre=1)            : %d" % len(arth_w2))
w("arthritis ever w2-w6 (cumulative)   : %d" % len(arth_cum))
w("w2 respondents with valid arthre    : %d" % len(w2_valid))

all3 = set(crp_pid.idauniq.dropna().astype(float)) if crp_pid.idauniq.dtype.kind == "f" \
    else set(crp_pid.idauniq.dropna())
# normalise to comparable type
w("idauniq dtype in panel: %s" % elsa["idauniq"].dtype)
panel_ids = set(elsa["idauniq"].dropna().tolist())
w("unique persons in harmonized panel : %d" % len(panel_ids))

pid3 = set(crp_pid["idauniq"].dropna().tolist())
pid3 = set(int(x) for x in pid3 if pd.notna(x))
arth_w2 = set(int(x) for x in arth_w2 if pd.notna(x))
arth_cum = set(int(x) for x in arth_cum if pd.notna(x))
w2_valid = set(int(x) for x in w2_valid if pd.notna(x))
panel_ids = set(int(x) for x in panel_ids if pd.notna(x))

w("\n>>> ATTRITION TABLE")
w("  ELSA harmonized respondents                    : %5d" % len(panel_ids))
w("  ... with >=3 valid CRP waves                   : %5d" % len(pid3 & panel_ids))
w("  ... with arthritis @w2                        : %5d" % len(arth_w2 & panel_ids))
w("  === CORE: arthritis@w2  x  >=3 CRP waves      : %5d" % len(arth_w2 & pid3 & panel_ids))
w("  === LOOSE: arthritis ever w2-w6 x >=3 CRP     : %5d" % len(arth_cum & pid3 & panel_ids))
w("  (sensitivity) arthritis@w2 x >=2 CRP waves    : %5d" %
  len(arth_w2 & set(int(x) for x in cnt[cnt >= 2].index.to_series()
                    .to_frame(index=False)["mergeid"].merge(keys, on="mergeid", how="left")["idauniq"]
                    .dropna().tolist()) & panel_ids))

core = arth_w2 & pid3 & panel_ids
loose = arth_cum & pid3 & panel_ids
w("\n  core cohort loss from arthritis-only: %d -> %d (%.1f%% retained)" %
  (len(arth_w2 & panel_ids), len(core),
   100.0 * len(core) / max(1, len(arth_w2 & panel_ids))))

# ------------------------------------------------------------------
sec("5. OUTCOMES in the core cohort: ADL/IADL w6-w9 and death")
e = elsa.copy()
e["idauniq"] = pd.to_numeric(e["idauniq"], errors="coerce")
e["adla"] = pd.to_numeric(e["adla"], errors="coerce")
e["iadla"] = pd.to_numeric(e["iadla"], errors="coerce")
e["adlfive"] = pd.to_numeric(e["adlfive"], errors="coerce")
e["iwstat"] = pd.to_numeric(e["iwstat"], errors="coerce")

for name, cohort in [("CORE (arth@w2 x >=3 CRP)", core),
                     ("LOOSE (arth ever w2-6 x >=3 CRP)", loose)]:
    sub = e[e.idauniq.isin(cohort)]
    w("\n--- %s : n=%d persons ---" % (name, len(cohort)))
    w("ADL/IADL availability by wave (non-missing adla / iadla):")
    w(sub[sub.wave.between(6, 9)].groupby("wave").agg(
        n_rows=("idauniq", "size"),
        n_adla=("adla", "count"), n_iadla=("iadla", "count"),
        mean_adla=("adla", "mean"), mean_iadla=("iadla", "mean"),
    ).round(3).to_string())
    nw = (sub[sub.wave.between(6, 9) & sub.adla.notna()]
          .groupby("idauniq")["wave"].nunique())
    w("  persons with >=2 ADL waves in w6-w9 : %d" % (nw >= 2).sum())
    w("  persons with >=3 ADL waves in w6-w9 : %d" % (nw >= 3).sum())
    # death: iwstat 5 = died this wave, 6 = died prev wave, 3 = died before w1
    dec = sub[sub.iwstat.isin([5, 6])]
    w("  death flags (iwstat 5/6) any wave  : %d rows / %d persons" %
      (len(dec), dec.idauniq.nunique()))
    dec_after = sub[(sub.wave >= 3) & (sub.iwstat.isin([5, 6]))]
    w("  death flags from w3 onward         : %d persons" % dec_after.idauniq.nunique())
    w("  iwstat==9 (unknown alive/dead)     : %d persons" %
      sub[sub.iwstat == 9].idauniq.nunique())
    # baseline characteristics
    b = sub[sub.wave == 2]
    w("  baseline w2: age mean=%s  female=%s  arthre=1 in %d" %
      (round(pd.to_numeric(b["agey"], errors="coerce").mean(), 1) if "agey" in b else "NA",
       round((pd.to_numeric(b["ragender"], errors="coerce") == 2).mean(), 3)
       if "ragender" in b else "NA",
       (pd.to_numeric(b["arthre"], errors="coerce") == 1).sum()))

# ------------------------------------------------------------------
sec("6. TEMPORAL OVERLAP CHECK (critical design issue)")
w("Exposure time points actually available : 2004(w2), 2008(w4), 2012(w6), 2016-19(w8/9)")
w("User's proposed outcome window          : w6-w9 = 2012, 2014, 2016, 2018")
w("=> CRP at 2012 and 2016-19 is CONCURRENT WITH / AFTER the outcome start.")
w("   A trajectory built on all 4 points cannot precede a w6-w9 ADL trajectory.")
w("\nCleaner alternative: exposure = 2004/2008/2012 (3 points), outcome = w7/w8/w9")
w("   (2014/2016/2018) ADL/IADL + death. Sample sizes:")
pid_w246 = set()
t246 = raw[(raw.crp_valid) & (raw.wave_raw.isin([2, 4, 6]))]
c3 = t246.groupby("mergeid").size()
ids3 = c3[c3 == 3].index.to_frame(index=False).rename(columns={"mergeid": "mergeid"})
ids3 = ids3.merge(keys, on="mergeid", how="left")
pid_w246 = set(int(x) for x in ids3.idauniq.dropna().tolist())
w("  persons with ALL 3 CRP (w2,w4,w6)                 : %d" % len(pid_w246))
w("  + arthritis@w2                                    : %d" %
  len(pid_w246 & arth_w2 & panel_ids))
sub_b = e[e.idauniq.isin(pid_w246 & arth_w2 & panel_ids)]
nw_b = (sub_b[sub_b.wave.between(7, 9) & sub_b.adla.notna()]
        .groupby("idauniq")["wave"].nunique())
w("  + >=2 ADL waves in w7-w9                          : %d" % (nw_b >= 2).sum())
w("  deaths w3+ in this cohort                         : %d" %
  sub_b[(sub_b.wave >= 3) & sub_b.iwstat.isin([5, 6])].idauniq.nunique())

# ------------------------------------------------------------------
sec("7. CHARLS replication (2011 / 2015 two CRP waves)")
c11 = pd.read_sql("SELECT id, newcrp AS crp2011 FROM w2011_blood", CC)
c15 = pd.read_sql("SELECT id, bl_crp AS crp2015 FROM w2015_blood", CC)
for nm, d, col in [("2011", c11, "crp2011"), ("2015", c15, "crp2015")]:
    d[col] = pd.to_numeric(d[col], errors="coerce")
    w("%s blood: %d rows, %d unique id, CRP non-null=%d, valid(0<crp<=10)=%d, >10=%d, <0=%d" %
      (nm, len(d), d.id.nunique(), d[col].notna().sum(),
       ((d[col] > 0) & (d[col] <= 10)).sum(), (d[col] > 10).sum(), (d[col] < 0).sum()))
m = c11.merge(c15, on="id", how="inner")
v11 = (m.crp2011 > 0) & (m.crp2011 <= 10)
v15 = (m.crp2015 > 0) & (m.crp2015 <= 10)
w("both waves present: %d persons; both CRP valid: %d persons" %
  (len(m), (v11 & v15).sum()))
w("mean CRP 2011=%s  2015=%s" %
  (round(m.loc[v11, "crp2011"].mean(), 3), round(m.loc[v15, "crp2015"].mean(), 3)))
m.to_csv(os.path.join(OUT, "charls_crp.csv"), index=False, encoding="utf-8-sig")

import pyreadr as _pr
ch = _pr.read_r(os.path.join(RDATA, "charls.RData"))["charls"]
w("\ncharls.RData: %d rows x %d cols; id dtype=%s" % (ch.shape[0], ch.shape[1], ch["id"].dtype))
ch["arthre"] = pd.to_numeric(ch["arthre"], errors="coerce")
w("arthritis (arthre=1) by wave: %s" %
  ch[ch.arthre == 1].groupby("wave").size().to_dict())
w("CHARLS id sample: %s ; blood id sample: %s" %
  (ch["id"].head(2).tolist(), c11["id"].head(2).tolist()))
w("CHARLS ADL cols: %s" % [c for c in ch.columns if "adl" in c.lower()][:12])
w("CHARLS death cols: %s" % [c for c in ch.columns if any(k in c.lower()
                                                           for k in ("died", "death", "iwstat", "dath"))][:12])

# ------------------------------------------------------------------
sec("8. Save core cohort ids")
pd.DataFrame({"idauniq": sorted(core)}).to_csv(
    os.path.join(OUT, "cohort_ids_core.csv"), index=False)
pd.DataFrame({"idauniq": sorted(loose)}).to_csv(
    os.path.join(OUT, "cohort_ids_loose.csv"), index=False)
w("saved: cohort_ids_core.csv (%d), cohort_ids_loose.csv (%d)" %
  (len(core), len(loose)))

LOG.close()
print("done")
