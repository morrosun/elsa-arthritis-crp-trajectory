# -*- coding: utf-8 -*-
"""Final feasibility audit: corrected death flags, gender, CHARLS ID alignment.
   Every section is wrapped so one failure cannot truncate the report."""
import os
import traceback
import psycopg2
import pandas as pd
import numpy as np
import pyreadr


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
LOG = open(os.path.join(OUT, "FINAL_AUDIT.txt"), "w", encoding="utf-8")


def w(*a):
    print(*a, file=LOG)
    LOG.flush()


def sec(t):
    w("\n" + "=" * 92)
    w("## " + t)
    w("=" * 92)


def safe(label, fn):
    try:
        return fn()
    except Exception as e:
        w("ERROR in %s: %r" % (label, e))
        w(traceback.format_exc()[-800:])
        return None


CE = _db("elsa")
CC = _db("charls")

crp = pd.read_csv(os.path.join(OUT, "crp_long.csv"), dtype={"mergeid": str})
wide = pd.read_csv(os.path.join(OUT, "analysis_wide.csv"), dtype={"mergeid": str})
w("loaded crp_long=%d rows ; analysis_wide=%d rows" % (len(crp), len(wide)))

# ---- IDs ----
v = crp[crp.crp_valid]
nw = v.groupby("mergeid").size()
pid3 = set(nw[nw >= 3].index.astype(str))
pid4 = set(nw[nw == 4].index.astype(str))
pid2 = set(nw[nw >= 2].index.astype(str))
v246 = v[v.wave_raw.isin([2, 4, 6])]
c246 = v246.groupby("mergeid").size()
pid_all246 = set(c246[c246 == 3].index.astype(str))
all_ids = set(wide.mergeid.astype(str))
arth2 = set(wide.loc[wide.arth2 == 1, "mergeid"].astype(str))
core = arth2 & pid3 & all_ids
strict = arth2 & pid4 & all_ids
optionB = arth2 & pid_all246 & all_ids

sec("1. Headline numbers (ELSA)")
w("arthritis @w2 (arthre=1)                       : %d" % len(arth2))
w("valid CRP >=3 waves                            : %d" % len(pid3 & all_ids))
w("valid CRP 4 waves                              : %d" % len(pid4 & all_ids))
w("CORE  = arth@w2 x >=3 CRP                      : %d" % len(core))
w("STRICT= arth@w2 x 4 CRP                        : %d" % len(strict))
w("OPTION B = arth@w2 x w2+w4+w6 CRP              : %d" % len(optionB))

# ---- PG iwstat + gender ----
sec("2. Death and sex (corrected, from PG not R panel)")
iwcols = ", ".join("p1.r%diwstat AS iw%d" % (i, i) for i in range(1, 10))
iw = pd.read_sql("""SELECT p1.mergeid, %s, k.radyear, k.ragender, k.rabyear
                    FROM gh_elsa_p1 p1 JOIN elsa_keys k USING (mergeid)""" % iwcols, CE)
iw["mergeid"] = iw.mergeid.astype(str)
for c in iw.columns:
    if c != "mergeid":
        iw[c] = pd.to_numeric(iw[c], errors="coerce")
w("PG iw9 distribution (all 19,802): %s" % iw.iw9.value_counts(dropna=False).to_dict())
w("PG ragender distribution        : %s" % iw.ragender.value_counts(dropna=False).to_dict())
w("PG radyear non-null             : %d" % iw.radyear.notna().sum())
w("R panel iwstat is ALL ZERO -> do NOT use elsa.RData$iwstat for death (panelisation bug)")


def death_report(name, ids):
    s = iw[iw.mergeid.isin(ids)].copy()
    dead_any = set()
    dead_after3 = set()
    dead_after7 = set()
    for i in range(1, 10):
        d = set(s.loc[s["iw%d" % i].isin([5, 6]), "mergeid"])
        dead_any |= d
        if i >= 3:
            dead_after3 |= d
        if i >= 7:
            dead_after7 |= d
    w("\n--- %s (n=%d) ---" % (name, len(ids)))
    w("  deaths (iwstat 5/6, any wave w1-w9) : %d (%.1f%%)" %
      (len(dead_any), 100.0 * len(dead_any) / max(1, len(ids))))
    w("  deaths flagged from w3 onward        : %d (%.1f%%)" %
      (len(dead_after3), 100.0 * len(dead_after3) / max(1, len(ids))))
    w("  deaths flagged from w7 onward        : %d" % len(dead_after7))
    w("  with verified death year (radyear)   : %d" % s.radyear.notna().sum())
    w("  alive-and-interviewed at w9 (iw9=1)  : %d" % (s.iw9 == 1).sum())
    w("  female (ragender=2)                  : %d (%.1f%%)" %
      ((s.ragender == 2).sum(), 100.0 * (s.ragender == 2).mean()))
    w("  mean birth year                      : %.1f" % s.rabyear.mean())
    return s


s_core = death_report("CORE", core)
s_strict = death_report("STRICT", strict)
s_B = death_report("OPTION B", optionB)

sec("3. Baseline age at w2 (R panel, agey is reliable)")
e = pyreadr.read_r(os.path.join(RDATA, "elsa.RData"))["elsa"]
e["mergeid"] = e.idauniqc.astype(str)
for name, ids in [("CORE", core), ("OPTION B", optionB)]:
    b = e[(e.wave == 2) & e.mergeid.isin(ids)]
    a = pd.to_numeric(b.agey, errors="coerce")
    w("%s: n=%d, age@w2 mean=%.1f SD=%.1f, range=%.0f-%.0f" %
      (name, len(b), a.mean(), a.std(), a.min(), a.max()))
    w("  baseline ADL sum (adltot6) = %.3f ; IADL (iadltot2_e) = %.3f" %
      (pd.to_numeric(b.adltot6, errors="coerce").mean(),
       pd.to_numeric(b.iadltot2_e, errors="coerce").mean()))

sec("4. Outcome trajectory (ADL 0-5 / IADL 0-3), w6-w9")
for name, ids in [("CORE", core), ("STRICT", strict), ("OPTION B", optionB)]:
    s = wide[wide.mergeid.isin(ids)]
    w("\n%s (n=%d):" % (name, len(ids)))
    for i in range(6, 10):
        w("  w%d  ADL n=%4d mean=%.3f   IADL n=%4d mean=%.3f" %
          (i, s["adl%d" % i].notna().sum(), s["adl%d" % i].mean(),
           s["iadl%d" % i].notna().sum(), s["iadl%d" % i].mean()))
    n_adl = s[["adl%d" % i for i in range(6, 10)]].notna().sum(axis=1)
    w("  >=2 / >=3 / 4 ADL waves: %d / %d / %d" %
      ((n_adl >= 2).sum(), (n_adl >= 3).sum(), (n_adl == 4).sum()))

sec("5. CHARLS replication: ID alignment")


def charls():
    c11 = pd.read_sql("SELECT id::text AS id, newcrp FROM w2011_blood", CC)
    c15 = pd.read_sql("SELECT id::text AS id, bl_crp FROM w2015_blood", CC)
    for nm, d in [("2011", c11), ("2015", c15)]:
        d["id"] = d.id.str.strip()
        w("%s: n=%d, id lengths=%s, sample=%s" %
          (nm, len(d), d.id.str.len().value_counts().to_dict(), d.id.head(3).tolist()))
    ch = pyreadr.read_r(os.path.join(RDATA, "charls.RData"))["charls"]
    ch["pid"] = ch.ID.astype(str).str.strip()
    panel = set(ch.pid)
    w("RData ID lengths=%s, sample=%s" %
      (ch.pid.str.len().value_counts().to_dict(), ch.pid.head(3).tolist()))

    def variants(s):
        return {
            "exact": s,
            "hh10+pad2": s.str[:10] + s.str[10:].str.zfill(2),
            "hh9+pad2": s.str[:9] + s.str[9:].str.zfill(2),
            "zpad12": s.str.zfill(12),
            "drop_last": s.str[:-1],
        }
    for nm, d in [("2011", c11), ("2015", c15)]:
        w("%s -> panel match counts: %s" %
          (nm, {k: len(set(v.dropna().astype(str)) & panel)
                for k, v in variants(d.id).items()}))
    a, b = set(c11.id), set(c15.id)
    w("2011 vs 2015 raw id overlap: %d" % len(a & b))
    for label, fn in [("exact", lambda x: x),
                      ("hh10+pad2", lambda x: x[:10] + x[10:].zfill(2)),
                      ("hh9+pad2", lambda x: x[:9] + x[9:].zfill(2))]:
        w("  %s transformed overlap: %d" %
          (label, len({fn(x) for x in a} & {fn(x) for x in b})))
    # extra columns in 2015 blood
    w("w2015_blood columns: %s" % list(pd.read_sql(
        "SELECT * FROM w2015_blood LIMIT 1", CC).columns))
    return c11, c15, ch


c11, c15, ch = safe("charls", charls)

sec("6. Save cohorts")
for nm, ids in [("core", core), ("strict", strict), ("optionB", optionB)]:
    pd.DataFrame({"mergeid": sorted(ids)}).to_csv(
        os.path.join(OUT, "cohort_%s.csv" % nm), index=False)
    w("cohort_%s.csv : %d persons" % (nm, len(ids)))
LOG.close()
print("done")
