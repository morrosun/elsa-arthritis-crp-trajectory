# -*- coding: utf-8 -*-
"""Group-based trajectory models (GBTM / latent-class linear mixed) for
   log-CRP and log-fibrinogen in each of the three competing designs.

Model: y_it | class k ~ N(b_k0 + b_k1*t [+ b_k2*t^2], sigma^2), pi_k
Estimated by EM with multiple starts; unbalanced panels handled by summing
each person's observed-time log-likelihood (FIML-like).
Outputs: out/TRAJ_RESULTS.txt + out/traj_<design>_<marker>_K<k>.csv
"""
import os
import itertools
import traceback
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import KMeans

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "TRAJ_RESULTS.txt"), "w", encoding="utf-8")


def w(*a):
    print(*a, file=LOG)
    LOG.flush()


def sec(t):
    w("\n" + "=" * 92)
    w("## " + t)
    w("=" * 92)


long = pd.read_csv(os.path.join(OUT, "exposure_long.csv"), dtype={"mergeid": str})
long["mergeid"] = long.mergeid.astype(str).str.strip()
base = pd.read_csv(os.path.join(OUT, "elsa_base.csv"), dtype={"mergeid": str})
base["mergeid"] = base.mergeid.astype(str).str.strip()


def fit_gbtm(y, t, pid, K, degree=1, n_init=8, max_iter=400, seed=11):
    """EM for group-based trajectory model with common residual variance."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones_like(t)] + [t ** j for j in range(1, degree + 1)])
    pid_code, pid_idx = np.unique(pid, return_inverse=True)
    N = len(pid_code)
    d = X.shape[1]
    npar = K * d + 1 + (K - 1)

    # per-person OLS coefficients for KMeans initialisation
    b0 = np.zeros(N)
    b1 = np.zeros(N)
    for i in range(N):
        m = pid_idx == i
        if m.sum() >= 2:
            co = np.polyfit(t[m], y[m], 1)
            b0[i], b1[i] = co[1], co[0]
        elif m.sum() == 1:
            b0[i] = y[m][0]
    Z = np.column_stack([b0, b1])
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)

    def design_for(P):
        """M-step given posterior P (n_obs x K)."""
        B = np.zeros((K, d))
        for k in range(K):
            W = P[:, k]
            A = X.T @ (W[:, None] * X) + 1e-8 * np.eye(d)
            B[k] = np.linalg.solve(A, X.T @ (W * y))
        pred = X @ B.T                       # n_obs x K
        r2 = (y[:, None] - pred) ** 2
        sig2 = max(0.02, float((P * r2).sum() / len(y)))
        return B, sig2

    def logjoint(B, sig2, logpi):
        pred = X @ B.T
        lp = -0.5 * np.log(2 * np.pi * sig2) - (y[:, None] - pred) ** 2 / (2 * sig2)
        # aggregate to person level
        L = np.zeros((N, K))
        np.add.at(L, pid_idx, lp)
        return L + logpi[None, :]

    best = None
    for start in range(n_init):
        if start == 0:
            km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(Z)
            hard = np.zeros((N, K))
            hard[np.arange(N), km.labels_] = 1.0
        elif start == 1:
            hard = rng.dirichlet(np.ones(K) * 0.6, size=N)
        else:
            km = KMeans(n_clusters=K, n_init=3,
                        random_state=seed + start).fit(Z + rng.normal(0, 0.5, Z.shape))
            hard = np.zeros((N, K))
            hard[np.arange(N), km.labels_] = 1.0
            hard = 0.75 * hard + 0.25 / K
        P = hard[pid_idx] * 0.9 + 0.1 / K
        B, sig2 = design_for(P)
        pi = P.sum(0)
        pi = pi / pi.sum()
        logpi = np.log(pi + 1e-12)
        ll_old = -np.inf
        for it in range(max_iter):
            L = logjoint(B, sig2, logpi)                     # (N, K) person level
            ll = float(logsumexp(L, axis=1).sum())
            P = np.exp(L - logsumexp(L, axis=1, keepdims=True))
            pi = P.mean(0)
            logpi = np.log(pi + 1e-12)
            B, sig2 = design_for(P[pid_idx])                # map back to observations
            if abs(ll - ll_old) < 1e-7:
                break
            ll_old = ll
        L = logjoint(B, sig2, logpi)
        ll = float(logsumexp(L, axis=1).sum())
        P = np.exp(L - logsumexp(L, axis=1, keepdims=True))
        if best is None or ll > best["ll"]:
            ent_numer = float(-(P * np.log(P + 1e-300)).sum())
            ent = 1 - ent_numer / (N * np.log(K))
            bic = -2 * ll + npar * np.log(N)
            aic = -2 * ll + 2 * npar
            best = dict(ll=ll, bic=bic, aic=aic, entropy=ent, B=B, sig2=sig2,
                        pi=pi, P=P, pid_code=pid_code, npar=npar, iters=it)
    return best


def describe(lab, fit):
    K = fit["B"].shape[0]
    assign = fit["P"].argmax(1)
    w("  %-28s K=%d  LL=%9.2f  BIC=%9.2f  AIC=%9.2f  entropy=%.3f  sigma2=%.3f" %
      (lab, K, fit["ll"], fit["bic"], fit["aic"], fit["entropy"], fit["sig2"]))
    # order classes by baseline level
    order = np.argsort(fit["B"][:, 0])
    for rank, k in enumerate(order, 1):
        m = assign == k
        appa = fit["P"][m, k].mean() if m.any() else np.nan
        b = fit["B"][k]
        shape = "rising" if b[1] > 0.06 else ("falling" if b[1] < -0.06 else "stable")
        w("     class %d  n=%4d (%5.1f%%)  APPA=%.3f  intercept=%7.3f  slope=%7.3f/4y  %s" %
          (rank, m.sum(), 100 * m.mean(), appa, b[0], b[1], shape))
    return assign, order


DESIGNS = {
    "A": dict(tset=[0, 1, 2, 3], min_waves=3,
              note="CORE: >=3 draws incl 2016-19; outcome w6-w9 (concurrent)"),
    "B": dict(tset=[0, 1, 2], min_waves=3,
              note="CLEAN: exactly w2/w4/w6; outcome w7-w9 (prospective)"),
    "C": dict(tset=[0, 1, 2, 3], min_waves=4,
              note="STRICT: all 4 draws; outcome w9 single time"),
}
MARKERS = {"crp": "logcrp", "fib": "logfib"}

for dkey, cfg in DESIGNS.items():
    ids = set(pd.read_csv(os.path.join(OUT, "cohort_design%s.csv" % dkey),
                          dtype={"mergeid": str}).mergeid.astype(str).str.strip())
    sec("DESIGN %s  (n=%d) | %s" % (dkey, len(ids), cfg["note"]))
    sub = long[(long.mergeid.isin(ids)) & (long.t.isin(cfg["tset"]))].copy()
    w("exposure rows available: %d" % len(sub))
    for mk, col in MARKERS.items():
        s = sub[sub[col].notna()].copy()
        cnt = s.groupby("mergeid").size()
        keep = set(cnt[cnt >= cfg["min_waves"]].index)
        s = s[s.mergeid.isin(keep)]
        w("\n--- marker %s : %d persons, %d obs ---" %
          (mk.upper(), s.mergeid.nunique(), len(s)))
        for K in [2, 3, 4, 5]:
            try:
                fit = fit_gbtm(s[col].to_numpy(float), s.t.to_numpy(float),
                               s.mergeid.to_numpy(), K, degree=1,
                               n_init=8, max_iter=400, seed=101 + K)
                assign, order = describe("%s K-sweep" % mk, fit)
                # save assignment for K=3 and K=4
                if K in (3, 4):
                    remap = {int(k): int(r) for r, k in enumerate(order, 1)}
                    out = pd.DataFrame({
                        "mergeid": fit["pid_code"],
                        "traj_class": [remap[int(a)] for a in assign],
                        "post_max": fit["P"].max(1),
                    })
                    for k in range(K):
                        out["p_class%d" % remap[k]] = fit["P"][:, k]
                    out["intercept"] = fit["B"][assign, 0]
                    out["slope"] = fit["B"][assign, 1]
                    out.to_csv(os.path.join(OUT, "traj_%s_%s_K%d.csv" % (dkey, mk, K)),
                               index=False, encoding="utf-8-sig")
            except Exception as ex:
                w("  ERROR K=%d %s: %r" % (K, mk, ex))
                w(traceback.format_exc()[-500:])
        # quadratic sensitivity for 4-point designs
        if cfg["min_waves"] == 4:
            try:
                f2 = fit_gbtm(s[col].to_numpy(float), s.t.to_numpy(float),
                              s.mergeid.to_numpy(), 3, degree=2,
                              n_init=5, max_iter=300, seed=77)
                w("  [quadratic sensitivity, K=3] BIC=%.2f (linear K=3 above for comparison) entropy=%.3f" %
                  (f2["bic"], f2["entropy"]))
            except Exception as ex:
                w("  quadratic ERROR %r" % ex)

sec("Notes")
w("t = 0/1/2/3 corresponds to 2004/05, 2008/09, 2012/13, 2016-19 (4-year steps).")
w("y = log(CRP mg/L) or log(fibrinogen g/L); negative sentinels and values >10 excluded.")
w("BIC = -2LL + npar*log(N_persons); lower is better. Entropy >0.7 and APPA >0.7 desirable.")
w("Common residual variance across classes is imposed for stability (LCA-like GBTM).")
w("Classes are ordered by baseline intercept; 'rising/falling' uses |slope|>0.06 log-units/4y.")
LOG.close()
print("done")
