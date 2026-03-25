"""
Figure 4 — FDR control and power of VD-T-Rex versus AD-T-Rex.

Sweeps SNR × L, comparing:
  - AD-T-Rex: explicit dummy augmentation (Python AD_LARS)
  - VD-T-Rex: virtual dummies (C++ TRexSelector from vd_selectors)

Produces the 2×3 panel FDP/TPP figure from Section 6.2.

Requirements:
    pip install numpy matplotlib joblib tqdm pandas scipy
    + vd_selectors (C++ extension, built via scikit-build-core)
    + AD_LARS.py and helpers.py in PYTHONPATH
"""
from __future__ import annotations

import os, sys, json
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed
from tqdm import tqdm
from contextlib import contextmanager
import pandas as pd

# ---------- algorithms ----------
from vd_selectors import (
    TRexSelector, TRexOptions, SolverType, CalibMode, VDDummyLaw,
)
from AD_LARS import AD_LARS

EPS = 1e-12


# ================================================================
# tqdm + joblib integration
# ================================================================
@contextmanager
def tqdm_joblib(tqdm_object):
    import joblib
    class _Cb(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *a, **kw):
            try: tqdm_object.update(n=self.batch_size)
            except Exception: pass
            return super().__call__(*a, **kw)
    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Cb
    try: yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_object.close()


# ================================================================
# Utilities
# ================================================================
def center_unitL2(X, eps=EPS, fortran=False):
    X = np.asarray(X, float)
    X -= X.mean(axis=0, keepdims=True)
    X /= np.linalg.norm(X, axis=0, keepdims=True).clip(min=eps)
    return np.asfortranarray(X) if fortran else X

def center(y):
    return np.asarray(y, float) - np.mean(y)

def fdr_tpp(selected, truth):
    sel = set(map(int, np.asarray(selected, int).ravel()))
    tru = set(map(int, np.asarray(truth, int).ravel()))
    tp, fp, R, P = len(sel & tru), len(sel - tru), len(sel), len(tru)
    return (fp / R if R else 0.0), (tp / P if P else 0.0)


# ================================================================
# Problem generation (random design)
# ================================================================
def make_problem(n=300, p=1000, s=10, beta_scale=1.0, rng=None):
    if rng is None: rng = np.random.default_rng()
    X = center_unitL2(rng.standard_normal((n, p)), fortran=True)
    support = np.sort(rng.choice(p, size=s, replace=False))
    beta = np.zeros(p)
    beta[support] = rng.choice([-1.0, 1.0], size=s) * beta_scale
    return X, beta, support

def sample_y(X, beta, snr, rng):
    signal = X @ beta
    sigv = max(float(np.var(signal)), 1e-20)
    sigma = np.sqrt(sigv / float(snr))
    return center(signal + sigma * rng.standard_normal(X.shape[0]))


# ================================================================
# T-Rex math helpers (for AD baseline)
# ================================================================
def phi_prime(p, T, Ld, phi_T, Phi):
    av = phi_T.sum(axis=0)
    rows = Phi > 0.5
    delta = phi_T[rows].sum(axis=0) if rows.any() else np.zeros(T)
    phi_mod, delta_mod = phi_T.copy(), delta.copy()
    if T > 1:
        phi_mod[:, 1:] -= phi_T[:, :-1]
        delta_mod[1:] -= delta[:-1]
    w = np.zeros(T)
    for t in range(T):
        denom = Ld - t
        if delta_mod[t] > EPS and denom > 0:
            w[t] = 1.0 - (p - av[t]) / (denom * delta_mod[t])
    return phi_mod @ w

def fdp_hat(V, Phi, PhiP):
    out = np.zeros_like(V)
    for i, v in enumerate(V):
        idx = Phi > v
        R = int(idx.sum())
        out[i] = 0.0 if R == 0 else min(1.0, float(np.sum(1.0 - PhiP[idx]) / R))
    return out

def select_vars(p, tFDR, FDP_mat, Phi_mat, V):
    feasible_T = np.where((FDP_mat <= tFDR).any(axis=1))[0]
    if feasible_T.size == 0:
        return np.array([], dtype=int), 0, float(V[-1]), 0
    T_select = int(feasible_T.max())
    R_mat = np.zeros_like(FDP_mat[:T_select+1], dtype=int)
    for t in range(T_select + 1):
        R_mat[t] = (Phi_mat[t][:, None] > V[None, :]).sum(axis=0)
    R_masked = np.where(FDP_mat[:T_select+1] <= tFDR, R_mat, -1)
    max_R = int(R_masked.max())
    locs = np.argwhere(R_masked == max_R)
    locs = locs[np.lexsort((locs[:, 0], locs[:, 1]))]
    t_idx, v_idx = locs[-1]
    v_star = float(V[v_idx])
    return np.flatnonzero(Phi_mat[t_idx] > v_star).astype(int), int(t_idx+1), v_star, max_R


# ================================================================
# AD-T-Rex (Python AD_LARS)
# ================================================================
def trex_ad_select(X, y, *, tFDR, K, num_dummies, Tmax, seed):
    """
    AD-T-Rex with persistent solvers: K solvers are created once (each with
    its own dummy realization) and advanced along the path, matching the C++
    run_early_stop_ behavior where the same dummy pool is used across T.
    """
    rng = np.random.default_rng(seed)
    n, p = X.shape
    V = np.append(np.arange(0.5, 1.0, 1.0 / K), 1.0 - EPS)

    # Create K solvers once, each with its own dummies
    solvers = []
    for _ in range(K):
        D = center_unitL2(rng.standard_normal((n, num_dummies)))
        XD = np.asfortranarray(np.hstack([X, D]))
        tl = AD_LARS(XD, np.asarray(y), int(num_dummies),
                     max_steps=min(XD.shape[0], XD.shape[1]),
                     normalize=False, eps=EPS, verbose=False)
        solvers.append(tl)

    # Advance all K solvers stride by stride, checking FDP at each T
    FDP_list, Phi_list = [], []

    for T in range(1, Tmax + 1):
        phi_acc = np.zeros(p)
        for tl in solvers:
            tl.run(T=T)  # warm-start: advances from previous state
            if T in tl.beta_dict:
                beta_t = tl.beta_dict[T]["beta"]
            else:
                beta_t = tl.beta_path[-1]
            phi_acc += (np.abs(beta_t[:p]) > EPS)
        Phi = phi_acc / K

        # Build phi_T_mat up to current T for Phi_prime computation
        Phi_list.append(Phi)
        phi_T_mat = np.column_stack(Phi_list)  # (p, T)

        PhiP = phi_prime(p, T, num_dummies, phi_T_mat, Phi)
        FDP = fdp_hat(V, Phi, PhiP)
        FDP_list.append(FDP)

        if FDP[-1] > tFDR:
            break

    FDP_mat = np.vstack(FDP_list)   # (T_final, |V|)
    Phi_mat_out = np.vstack([ph.reshape(1, -1) for ph in Phi_list])  # (T_final, p)
    return select_vars(p, tFDR, FDP_mat, Phi_mat_out, V)


# ================================================================
# VD-T-Rex (C++ vd_selectors)
# ================================================================
def trex_vd_select(X, y, *, tFDR, K, L_factor, Tmax, seed):
    opt = TRexOptions()
    opt.tFDR = float(tFDR)
    opt.K = int(K)
    opt.L_factor = int(L_factor)
    opt.T_stop = int(Tmax)
    opt.seed = int(seed)
    opt.verbose = False
    opt.solver = SolverType.LARS
    opt.calib = CalibMode.CalibrateT
    opt.dummy_law = VDDummyLaw.Spherical
    opt.n_threads = 1
    opt.max_stale_strides = 999
    opt.stride_width = 1
    opt.posthoc_mode = False

    sel = TRexSelector(opt)
    res = sel.run(np.asfortranarray(X), np.asarray(y))
    selected = np.asarray(res.selected_var, dtype=int)
    T_star = int(res.T_stop)
    v_star = float(res.v_thresh)
    return selected, T_star, v_star, len(selected)


# ================================================================
# One Monte Carlo replicate
# ================================================================
def one_rep(rep, snr, alpha, K, L, Tmax, seed0, *, n, p, s, beta_scale):
    mix = (rep + 1) * 10_000_019 + int(1_000_003 * snr) + (L + 1) * 97
    rng = np.random.default_rng(int(seed0 + mix))
    X, beta, support = make_problem(n=n, p=p, s=s, beta_scale=beta_scale, rng=rng)
    y = sample_y(X, beta, float(snr), rng)

    sel_ad, *_ = trex_ad_select(
        X, y, tFDR=alpha, K=K, num_dummies=X.shape[1]*L, Tmax=Tmax,
        seed=int(rng.integers(2**31)),
    )
    sel_vd, *_ = trex_vd_select(
        X, y, tFDR=alpha, K=K, L_factor=L, Tmax=Tmax,
        seed=int(rng.integers(2**31)),
    )
    fdr_ad, tpp_ad = fdr_tpp(sel_ad, support)
    fdr_vd, tpp_vd = fdr_tpp(sel_vd, support)
    return {"fdp_ad": fdr_ad, "tpp_ad": tpp_ad, "fdp_vd": fdr_vd, "tpp_vd": tpp_vd}


# ================================================================
# Sweep
# ================================================================
def sweep(snrs, Ls, *, alpha=0.1, reps=50, K=20, Tmax=20, seed0=0,
          n_jobs=-1, n=300, p=1000, s=10, beta_scale=1.0):
    rows = []
    total = len(Ls) * len(snrs) * reps
    with tqdm_joblib(tqdm(total=total, desc="MC tasks")):
        for L in Ls:
            for snr in snrs:
                res = Parallel(n_jobs=n_jobs, backend="loky")(
                    delayed(one_rep)(
                        r, float(snr), float(alpha), int(K), int(L), int(Tmax), int(seed0),
                        n=n, p=p, s=s, beta_scale=beta_scale,
                    ) for r in range(reps)
                )
                fdp_ad = np.mean([d["fdp_ad"] for d in res])
                tpp_ad = np.mean([d["tpp_ad"] for d in res])
                fdp_vd = np.mean([d["fdp_vd"] for d in res])
                tpp_vd = np.mean([d["tpp_vd"] for d in res])
                rows.append({"L": L, "snr": snr, "method": "AD-T-Rex", "fdp": fdp_ad, "tpp": tpp_ad})
                rows.append({"L": L, "snr": snr, "method": "VD-T-Rex", "fdp": fdp_vd, "tpp": tpp_vd})
    return pd.DataFrame(rows)


# ================================================================
# Plotting
# ================================================================
def plot_results(df, alpha, outdir):
    colors = {"AD-T-Rex": "#8b0000", "VD-T-Rex": "#0b7a77"}
    styles = {1: "-", 5: "--", 10: ":", 20: "-.", 30: (0,(3,1,1,1)), 40: (0,(5,2))}
    snrs = sorted(df["snr"].unique())

    for metric, ylabel, fname in [("fdp", "FDP", "fdp_vs_snr"), ("tpp", "TPP", "tpp_vs_snr")]:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        for (method, L), g in df.groupby(["method", "L"], sort=True):
            g = g.sort_values("snr")
            ax.plot(g["snr"], g[metric], color=colors.get(method, "k"),
                    linestyle=styles.get(int(L), "-"), linewidth=2.5,
                    label=f"{method}, L={int(L)}p")
        if metric == "fdp":
            ax.axhline(alpha, color="k", ls=":", lw=1.8)
        ax.set_xticks(snrs); ax.set_xlabel("SNR"); ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.02); ax.grid(True, ls="--", alpha=0.35)
        ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
        fig.savefig(outdir / f"fig4_{fname}.pdf", bbox_inches="tight")
        fig.savefig(outdir / f"fig4_{fname}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    outdir = Path("results") / f"fig4_fdr_control"
    (outdir / "data").mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    snrs = (0.1, 0.5, 1.0, 3.0, 5.0 )
    Ls = (10,)
    alpha = 0.1
    reps = 100
    K = 20
    Tmax = 50
    seed0 = 1231
    n, p, s = 300, 1000, 10

    df = sweep(snrs=snrs, Ls=Ls, alpha=alpha, reps=reps, K=K, Tmax=Tmax,
               seed0=seed0, n_jobs=-1, n=n, p=p, s=s)

    df.to_csv(outdir / "data" / "summary.csv", index=False)
    meta = dict(snrs=list(snrs), Ls=list(Ls), alpha=alpha, reps=reps,
                K=K, Tmax=Tmax, seed0=seed0, n=n, p=p, s=s)
    json.dump(meta, open(outdir / "data" / "meta.json", "w"), indent=2)

    plot_results(df, alpha=alpha, outdir=outdir / "plots")
    print(f"Saved to {outdir}")