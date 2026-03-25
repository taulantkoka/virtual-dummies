"""
Figure 6 — Finite-sample effect of Gaussian norm fluctuations in T-Rex.

Compares spherical vs Gaussian dummy base laws across a grid of (alpha, T),
producing heatmaps of ΔFDP and ΔTPP (= Spherical − Gaussian).

Produces the 2×3 heatmap figure from Section 6.4.

Requirements:
    pip install numpy matplotlib joblib tqdm
    + vd_selectors (C++ extension, built via scikit-build-core)
"""
from __future__ import annotations

import os, json, pathlib, datetime
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import matplotlib.pyplot as plt

from vd_selectors import VD_LARS, VDOptions, VDDummyLaw

EPS = 1e-12


# ================================================================
# Utilities
# ================================================================
def center_unitL2(X, eps=EPS):
    X = X - X.mean(axis=0, keepdims=True)
    return X / np.linalg.norm(X, axis=0, keepdims=True).clip(min=eps)

def center(y):
    return y - y.mean()

def compute_sigma_for_snr(X, beta, SNR, eps=EPS):
    return np.linalg.norm(X @ beta) / max(SNR * np.sqrt(X.shape[0]), eps)


# ================================================================
# Core: one VD-LARS experiment
# ================================================================
def one_vd_lars_path(Xc, yc, *, T_stop, num_dummies, seed, dummy_law,
                     eps=EPS, max_vd_proj=None):
    """Run one VD-LARS path, return phi_T (p × T_stop): active-set masks per T."""
    n, p = Xc.shape

    opt = VDOptions()
    opt.T_stop = int(T_stop)
    opt.max_vd_proj = int(min(n, max_vd_proj or max(32, n)))
    opt.eps = float(eps)
    opt.standardize = False
    opt.seed = int(seed)
    opt.dummy_law = dummy_law

    Xf = np.asfortranarray(Xc, dtype=np.float64)
    yf = np.asarray(yc, dtype=np.float64)
    m = VD_LARS(Xf, yf, int(num_dummies), opt)

    phi_cols = []
    for t in range(1, T_stop + 1):
        m.run(int(t))
        mask = np.zeros(p, dtype=np.float64)
        for af in m.active_features():
            kind, j = af if isinstance(af, tuple) else (None, None)
            if kind is None:
                kind = getattr(af, "kind", None)
                j = getattr(af, "index", None)
            kind_str = str(kind).lower() if kind is not None else ""
            if "dum" not in kind_str and j is not None:
                j = int(j)
                if 0 <= j < p:
                    mask[j] = 1.0
        phi_cols.append(mask)

    return np.stack(phi_cols, axis=1)   # (p, T_stop)


def random_experiments_parallel(Xc, yc, *, K, T_stop, num_dummies, seeds_k,
                                dummy_law, n_jobs_k=-1, eps=EPS, max_vd_proj=None):
    """Run K independent VD-LARS experiments, return averaged phi_T_mat (p × T_stop)."""
    def _one(k):
        return one_vd_lars_path(
            Xc, yc, T_stop=T_stop, num_dummies=num_dummies,
            seed=int(seeds_k[k]), dummy_law=dummy_law, eps=eps,
            max_vd_proj=max_vd_proj,
        )
    mats = Parallel(n_jobs=n_jobs_k, backend="loky")(delayed(_one)(k) for k in range(K))
    return np.mean(np.stack(mats), axis=0)


# ================================================================
# FDP_hat machinery
# ================================================================
def Phi_prime_fun(p, T_stop, num_dummies, phi_T_mat, Phi, eps=EPS):
    av = phi_T_mat.sum(axis=0)
    rows_gt = Phi > 0.5
    delta_av = phi_T_mat[rows_gt].sum(axis=0) if rows_gt.any() else np.zeros(T_stop)
    delta_mod, phi_mod = delta_av.copy(), phi_T_mat.copy()
    if T_stop > 1:
        delta_mod[1:] = delta_av[1:] - delta_av[:-1]
        phi_mod[:, 1:] = phi_T_mat[:, 1:] - phi_T_mat[:, :-1]
    t = np.arange(1, T_stop + 1)
    denom = num_dummies - t + 1
    w = np.zeros(T_stop)
    valid = (delta_mod > eps) & (denom > 0)
    if valid.any():
        w[valid] = 1.0 - (p - av[valid]) / (denom[valid] * delta_mod[valid])
    return (phi_mod @ w).ravel()

def fdp_hat_row(V, Phi, Phi_prime, eps=EPS):
    out = np.full(V.size, np.nan)
    for i, v in enumerate(V):
        idx = Phi > v
        R = int(idx.sum())
        out[i] = 0.0 if R == 0 else min(1.0, float(np.sum((1.0 - Phi_prime)[idx]) / R))
    return out

def empirical_fdp_tpp_R(phi_p_by_T, V, truth):
    p, T_stop = phi_p_by_T.shape
    FDP = np.zeros((T_stop, V.size))
    TPP = np.zeros((T_stop, V.size))
    R   = np.zeros((T_stop, V.size), int)
    for t in range(T_stop):
        Phi_t = phi_p_by_T[:, t]
        for vi, v in enumerate(V):
            sel = set(np.where(Phi_t > v)[0])
            r = len(sel); R[t, vi] = r
            FDP[t, vi] = len(sel - truth) / r if r else 0.0
            TPP[t, vi] = len(sel & truth) / len(truth) if truth else 0.0
    return FDP, TPP, R

def vstar_indices(FDP_hat_mat, alpha_grid):
    T, Vn = FDP_hat_mat.shape
    A = alpha_grid.size
    v_idx = np.full((A, T), -1, dtype=int)
    for t in range(T):
        feasible = FDP_hat_mat[t][None, :] <= alpha_grid[:, None]
        has = feasible.any(axis=1)
        v_idx[:, t] = np.where(has, feasible.argmax(axis=1), -1)
    return v_idx


# ================================================================
# One MC replicate: both laws on same (X, y)
# ================================================================
def one_mc_replicate(*, n, p, s, SNR, K, T_stop, num_dummies, alpha_grid,
                     seed, eps=EPS, n_jobs_k=-1, max_vd_proj=None):
    rng = np.random.default_rng(seed)
    Xc = center_unitL2(rng.standard_normal((n, p)), eps=eps)
    beta = np.zeros(p); beta[:s] = 2.0
    sigma = compute_sigma_for_snr(Xc, beta, SNR, eps)
    yc = center(Xc @ beta + sigma * rng.standard_normal(n))
    V = np.append(np.arange(0.5, 1.0, 1.0 / K), 1.0 - eps)
    seeds_k = rng.integers(0, 2**31 - 1, size=K, dtype=np.int64)
    truth = set(range(s))

    def _for_law(law):
        phi = random_experiments_parallel(
            Xc, yc, K=K, T_stop=T_stop, num_dummies=num_dummies,
            seeds_k=seeds_k, dummy_law=law, n_jobs_k=n_jobs_k, eps=eps,
            max_vd_proj=max_vd_proj,
        )
        FDPh = np.zeros((T_stop, V.size))
        for t in range(1, T_stop + 1):
            Phi_t = phi[:, t-1]
            PhiP = Phi_prime_fun(p, t, num_dummies, phi[:, :t], Phi_t, eps)
            FDPh[t-1] = fdp_hat_row(V, Phi_t, PhiP, eps)
        FDP_true, TPP_mat, R_mat = empirical_fdp_tpp_R(phi, V, truth)
        vi = vstar_indices(FDPh, alpha_grid)
        A = alpha_grid.size
        FDP_s = np.zeros((A, T_stop)); TPP_s = np.zeros((A, T_stop))
        for a in range(A):
            for t in range(T_stop):
                j = int(vi[a, t])
                if j >= 0:
                    FDP_s[a, t] = FDP_true[t, j]; TPP_s[a, t] = TPP_mat[t, j]
        return dict(FDP=FDP_s, TPP=TPP_s)

    out_sph = _for_law(VDDummyLaw.Spherical)
    out_gau = _for_law(VDDummyLaw.Gaussian)
    return dict(sph=out_sph, gau=out_gau)


# ================================================================
# MC driver
# ================================================================
def run_mc(*, M, n, p, s, SNR, K, T_stop, num_dummies, alpha_grid,
           seed=0, n_jobs_mc=1, n_jobs_k=-1, eps=EPS, max_vd_proj=None):
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31 - 1, size=M, dtype=np.int64)

    def _wrapped(sd):
        return one_mc_replicate(
            n=n, p=p, s=s, SNR=SNR, K=K, T_stop=T_stop, num_dummies=num_dummies,
            alpha_grid=alpha_grid, seed=int(sd), eps=eps, n_jobs_k=n_jobs_k,
            max_vd_proj=max_vd_proj,
        )

    if n_jobs_mc == 1:
        results = [_wrapped(int(sd)) for sd in tqdm(seeds, desc="MC replicates")]
    else:
        results = Parallel(n_jobs=n_jobs_mc, backend="loky")(
            delayed(_wrapped)(int(sd)) for sd in tqdm(seeds, desc="MC replicates"))

    A, T = alpha_grid.size, T_stop
    acc = {k: {"FDP": np.zeros((A, T)), "TPP": np.zeros((A, T))} for k in ("sph", "gau")}
    for out in results:
        for k in ("sph", "gau"):
            acc[k]["FDP"] += out[k]["FDP"]
            acc[k]["TPP"] += out[k]["TPP"]
    for k in ("sph", "gau"):
        acc[k]["FDP"] /= M; acc[k]["TPP"] /= M

    return dict(alpha=alpha_grid, Ts=np.arange(1, T_stop+1), sph=acc["sph"], gau=acc["gau"])


# ================================================================
# Plotting: heatmaps of Δ = Sph − Gau
# ================================================================
def plot_heatmaps(results, Ls_labels, outdir):
    """results: list of dicts (one per L), Ls_labels: list of str."""
    n_L = len(results)
    fig, axes = plt.subplots(2, n_L, figsize=(5.5 * n_L, 8), squeeze=False)

    for col, (res, L_lbl) in enumerate(zip(results, Ls_labels)):
        alpha = res["alpha"]; Ts = res["Ts"]
        dFDP = (res["sph"]["FDP"] - res["gau"]["FDP"]) * 100   # pp
        dTPP = (res["sph"]["TPP"] - res["gau"]["TPP"]) * 100

        for row, (Z, title) in enumerate([(dFDP, rf"$\Delta$FDP ({L_lbl})"),
                                           (dTPP, rf"$\Delta$TPP ({L_lbl})")]):
            ax = axes[row, col]
            vmax = max(abs(Z.min()), abs(Z.max()), 1.0)
            im = ax.imshow(Z, aspect="auto", origin="lower",
                           extent=[Ts[0]-0.5, Ts[-1]+0.5, alpha[0], alpha[-1]],
                           cmap="Reds", vmin=0, vmax=5)
            ax.set_xlabel("$T$"); ax.set_ylabel(r"$\alpha$")
            ax.set_title(title)
            plt.colorbar(im, ax=ax, label="pp")

    fig.tight_layout()
    fig.savefig(outdir / "fig6_spherical_vs_gaussian.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fig6_spherical_vs_gaussian.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# Main
# ================================================================
def run_and_save(*, out_root="results", M=1000, n=300, p=10000, s=10, SNR=1.0,
                 seed=123, K=20, T_stop=20, L_list=(5, 10, 20),
                 alpha_grid=None, n_jobs_mc=-1, n_jobs_k=1):
    if alpha_grid is None:
        alpha_grid = np.linspace(0.05, 0.30, 30)
    alpha_grid = np.asarray(alpha_grid, float)

    outdir = pathlib.Path(out_root) / f"fig6_sph_vs_gau"
    outdir.mkdir(parents=True, exist_ok=True)

    all_results = []
    Ls_labels = []

    for L in L_list:
        num_dummies = int(L * p)
        print(f"\n{'='*60}\nL = {L}p = {num_dummies}\n{'='*60}")
        res = run_mc(
            M=M, n=n, p=p, s=s, SNR=SNR, K=K, T_stop=T_stop,
            num_dummies=num_dummies, alpha_grid=alpha_grid, seed=seed,
            n_jobs_mc=n_jobs_mc, n_jobs_k=n_jobs_k,
        )
        all_results.append(res)
        Ls_labels.append(f"$L={L}p$")

        np.savez_compressed(
            outdir / f"surfaces_L{L}p.npz",
            alpha=res["alpha"], Ts=res["Ts"],
            FDP_sph=res["sph"]["FDP"], TPP_sph=res["sph"]["TPP"],
            FDP_gau=res["gau"]["FDP"], TPP_gau=res["gau"]["TPP"],
        )

    meta = dict(M=M, n=n, p=p, s=s, SNR=SNR, K=K, T_stop=T_stop,
                L_list=list(L_list), seed=seed,
                alpha=list(map(float, alpha_grid)))
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

    plot_heatmaps(all_results, Ls_labels, outdir)
    print(f"\nSaved to: {outdir}")
    return str(outdir)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    run_and_save(
        out_root="results",
        M=1000,
        n=300,
        p=10000,
        s=10,
        SNR=1.0,
        seed=123,
        K=20,
        T_stop=20,
        L_list=(5, 10, 20),
        alpha_grid=np.linspace(0.05, 0.30, 30),
        n_jobs_mc=-1,
        n_jobs_k=1,
    )
