"""
Figure 3 — Distributional equivalence of VD-LARS and AD-LARS.

Compares order-statistic trajectories, ECDFs, and QQ plots of absolute
dummy–residual correlations between:
  - AD-LARS: explicit dummy augmentation (Python AD_LARS)
  - VD-LARS: virtual dummies (C++ vd_selectors)

Produces the 2×4 panel figure from Section 6.1.

Requirements:
    pip install numpy matplotlib joblib tqdm scipy
    + vd_selectors (C++ extension, built via scikit-build-core)
    + AD_LARS_new.py and helpers.py in PYTHONPATH
"""
from __future__ import annotations

import os, sys, json, pathlib
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from joblib import Parallel, delayed
from tqdm import tqdm
from matplotlib.lines import Line2D
import matplotlib as mpl

# ---------- algorithms ----------
from AD_LARS import AD_LARS                                      # AD baseline
from vd_selectors import VD_LARS, VDOptions, VDDummyLaw            # VD (C++)

EPS = 1e-12

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "TeX Gyre Termes", "Nimbus Roman"],
    "font.size": 14,
    "mathtext.fontset": "cm",
    "axes.titlesize": 16, "axes.labelsize": 16,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "legend.fontsize": 14, "axes.unicode_minus": False,
})


# ================================================================
# Helpers
# ================================================================
def center_unitL2(X, eps=EPS):
    X = np.asarray(X, float)
    X -= X.mean(axis=0, keepdims=True)
    X /= np.linalg.norm(X, axis=0, keepdims=True).clip(min=eps)
    return np.asfortranarray(X)

def center(y):
    y = np.asarray(y, float)
    return y - y.mean()

def make_y_snr(X, beta, snr, rng):
    signal = X @ beta
    sigv = max(float(np.var(signal)), 1e-20)
    sigma = np.sqrt(sigv / float(snr))
    return center(signal + sigma * rng.standard_normal(X.shape[0]))

def make_dummies_unitL2(n, Lp, rng, eps=EPS):
    D = rng.standard_normal((n, Lp))
    D -= D.mean(axis=0, keepdims=True)
    D /= np.linalg.norm(D, axis=0, keepdims=True).clip(min=eps)
    return np.asfortranarray(D)

def top_m_desc(x, m):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.full(m, np.nan)
    if x.size <= m:
        out = np.sort(x)[::-1]
        return np.concatenate([out, np.full(m - out.size, np.nan)]) if out.size < m else out
    return np.sort(np.partition(x, -m)[-m:])[::-1]

def make_fixed_problem(n, p, s, snr, seed):
    rng = np.random.default_rng(seed)
    X = center_unitL2(rng.standard_normal((n, p)))
    support = np.sort(rng.choice(p, size=s, replace=False))
    beta = np.zeros(p)
    beta[support] = rng.choice([-1.0, 1.0], size=s)
    y = make_y_snr(X, beta, snr, rng)
    return X, y, support

def subsample_abs_corr(x, m, rng):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.full(m, np.nan)
    if x.size <= m:
        return np.concatenate([x, np.full(m - x.size, np.nan)]) if x.size < m else x.copy()
    return x[rng.choice(x.size, size=m, replace=False)]

def ecdf_vals(x):
    x = np.sort(np.asarray(x, float)[np.isfinite(np.asarray(x, float))])
    if x.size == 0:
        return np.array([0.0]), np.array([0.0])
    return x, np.arange(1, x.size + 1) / x.size

def qq_xy(x, y, q):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.zeros_like(q), np.zeros_like(q)
    return np.quantile(x, q), np.quantile(y, q)


# ================================================================
# One replicate
# ================================================================
@dataclass
class RepTraj:
    traj_ad: np.ndarray             # (Tmax, nranks)
    traj_vd: np.ndarray             # (Tmax, nranks)
    corr_ad: dict[int, np.ndarray]  # T -> (m_corr,)
    corr_vd: dict[int, np.ndarray]  # T -> (m_corr,)

def one_rep_traj(
    seed, *, X, y, p, Lp, Tmax, ranks, m_top, T_show, m_corr,
    vd_max_proj=None,
):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    nr = len(ranks)

    # ---------- AD: explicit dummies ----------
    D = make_dummies_unitL2(n, Lp, rng)
    XD = np.hstack([X, D])
    tl = AD_LARS(XD, y, num_dummies=Lp, normalize=False, eps=EPS, verbose=False)

    traj_ad = np.full((Tmax, nr), np.nan)
    corr_ad = {}

    for T in range(1, Tmax + 1):
        tl.run(T=T)
        r = np.asarray(tl.residuals, float)
        entered_local = np.array([j - p for j in tl.selected_dummies if j >= p], dtype=int)
        mask_unused = np.ones(Lp, dtype=bool)
        valid = entered_local[(entered_local >= 0) & (entered_local < Lp)]
        if valid.size:
            mask_unused[valid] = False
        rho_unused = np.abs(D.T @ r)[mask_unused]
        os_vals = top_m_desc(rho_unused, m_top)
        for j, rk in enumerate(ranks):
            traj_ad[T - 1, j] = os_vals[rk - 1]
        if T in T_show:
            corr_ad[T] = subsample_abs_corr(rho_unused, m_corr, rng)

    # ---------- VD: virtual dummies (C++) ----------
    vd_seed = int(rng.integers(0, 2**31 - 1))
    opt = VDOptions()
    opt.T_stop = int(Tmax)
    opt.max_vd_proj = int(min(n, vd_max_proj or max(32, n)))
    opt.eps = float(EPS)
    opt.standardize = False
    opt.seed = vd_seed
    opt.dummy_law = VDDummyLaw.Spherical

    Xf = np.asfortranarray(X, dtype=np.float64)
    yf = np.asarray(y, dtype=np.float64)
    vd = VD_LARS(Xf, yf, int(Lp), opt)

    traj_vd = np.full((Tmax, nr), np.nan)
    corr_vd = {}

    for T in range(1, Tmax + 1):
        vd.run(T)
        realized = np.asarray(vd.is_dummy_realized(), dtype=bool)
        vd_corr = np.asarray(vd.vd_corr(), dtype=float)
        rho_unused = np.abs(vd_corr[~realized])
        os_vals = top_m_desc(rho_unused, m_top)
        for j, rk in enumerate(ranks):
            traj_vd[T - 1, j] = os_vals[rk - 1]
        if T in T_show:
            corr_vd[T] = subsample_abs_corr(rho_unused, m_corr, rng)

    return RepTraj(traj_ad=traj_ad, traj_vd=traj_vd, corr_ad=corr_ad, corr_vd=corr_vd)


# ================================================================
# Plotting
# ================================================================
def summarize_bands(arr, lo, hi):
    return np.nanmedian(arr, axis=0), np.nanquantile(arr, lo, axis=0), np.nanquantile(arr, hi, axis=0)

def plot_combined(
    Tgrid, traj_ad, traj_vd, ranks,
    corr_ad_all, corr_vd_all, T_ecdf, T_qq,
    outdir, *, band=(0.01, 0.99),
):
    outdir.mkdir(parents=True, exist_ok=True)
    lo, hi = band
    nr = len(ranks)
    nc = nr

    if len(T_ecdf) + len(T_qq) != nc:
        raise ValueError(f"Need len(T_ecdf)+len(T_qq)=={nc}")

    fig = plt.figure(figsize=(5.2 * nc, 7.2))
    gs = fig.add_gridspec(2, nc, wspace=0.25, hspace=0.35)

    # Shared y-limits for trajectory panels
    ad_lo = np.nanquantile(traj_ad, lo, axis=0)
    ad_hi = np.nanquantile(traj_ad, hi, axis=0)
    vd_lo = np.nanquantile(traj_vd, lo, axis=0)
    vd_hi = np.nanquantile(traj_vd, hi, axis=0)
    ymin = float(np.nanmin([ad_lo.min(), vd_lo.min()]))
    ymax = float(np.nanmax([ad_hi.max(), vd_hi.max()]))
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        pad = 0.03 * (ymax - ymin)
        ymin -= pad; ymax += pad

    # Top row: trajectories
    for j, rk in enumerate(ranks):
        ax = fig.add_subplot(gs[0, j])
        ad = traj_ad[:, :, j]; vd = traj_vd[:, :, j]
        ad_med, ad_qlo, ad_qhi = summarize_bands(ad, lo, hi)
        vd_med, vd_qlo, vd_qhi = summarize_bands(vd, lo, hi)

        ax.plot(Tgrid, vd_med, lw=2.2, color="tab:blue")
        ax.plot(Tgrid, vd_qlo, lw=1.4, color="tab:blue", ls=":", alpha=0.95)
        ax.plot(Tgrid, vd_qhi, lw=1.4, color="tab:blue", ls=":", alpha=0.95)
        ax.plot(Tgrid, ad_med, lw=2.2, color="tab:red", ls="--")
        ax.plot(Tgrid, ad_qlo, lw=1.4, color="tab:red", ls=":", alpha=0.95)
        ax.plot(Tgrid, ad_qhi, lw=1.4, color="tab:red", ls=":", alpha=0.95)

        ax.set_title(rf"Order stat rank-{rk}")
        ax.set_xlabel(r"$T$"); ax.set_ylabel(rf"$(|\rho|)_{{{rk}}}$")
        ax.set_xlim(Tgrid[0], Tgrid[-1])
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.35, ls=":")

    # Bottom-left: ECDFs
    for j, T in enumerate(T_ecdf):
        ax = fig.add_subplot(gs[1, j])
        xs_ad, ys_ad = ecdf_vals(corr_ad_all.get(T, np.array([])))
        xs_vd, ys_vd = ecdf_vals(corr_vd_all.get(T, np.array([])))
        ax.step(xs_vd, ys_vd, where="post", color="tab:blue", lw=2.0)
        ax.step(xs_ad, ys_ad, where="post", color="tab:red", lw=2.0, ls="--")
        ax.set_title(rf"$T={T}$"); ax.set_xlabel(r"$|\rho|$"); ax.set_ylabel("ECDF")
        ax.grid(True, alpha=0.35, ls=":")

    # Bottom-right: QQ plots
    qgrid = np.linspace(0.01, 0.99, 200)
    for j, T in enumerate(T_qq):
        ax = fig.add_subplot(gs[1, len(T_ecdf) + j])
        x_ad = corr_ad_all.get(T, np.array([])); x_vd = corr_vd_all.get(T, np.array([]))
        qx, qy = qq_xy(x_ad, x_vd, qgrid)
        lim = float(max(np.nanmax(qx), np.nanmax(qy), 1e-12))
        ax.plot(qx, qy, ".", ms=2.2, color="k")
        ax.plot([0, lim], [0, lim], "k--", lw=1.0)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_title(rf"$T={T}$")
        ax.set_xlabel(r"$|\rho|_{\mathrm{AD}}$ quantiles")
        ax.set_ylabel(r"$|\rho|_{\mathrm{VD}}$ quantiles")
        ax.grid(True, alpha=0.35, ls=":")

    # Legend
    band_txt = f"{int(lo*100)}\u2013{int(hi*100)}%"
    handles = [
        Line2D([0], [0], color="tab:blue", lw=2.2, ls="-"),
        Line2D([0], [0], color="tab:blue", lw=1.4, ls=":"),
        Line2D([0], [0], color="tab:red",  lw=2.2, ls="--"),
        Line2D([0], [0], color="tab:red",  lw=1.4, ls=":"),
    ]
    labels = ["VD median", f"VD {band_txt} bounds", "AD median", f"AD {band_txt} bounds"]
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.01), fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outdir / "fig3_distributional_equivalence.pdf", bbox_inches="tight")
    fig.savefig(outdir / "fig3_distributional_equivalence.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


# ================================================================
# Driver
# ================================================================
def run(*, outdir, n=300, p=1000, s=10, snr=1.0, L=5, Tmax=20, K=2000,
        m_top=200, ranks=[1, 5, 20, 50], band=(0.01, 0.99), seed=123,
        n_jobs=-1, T_show=None, m_corr=2000):

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    Lp = L * p
    X, y, support = make_fixed_problem(n, p, s, snr, seed)

    if T_show is None:
        T_show = sorted({1, Tmax})

    rng = np.random.default_rng(seed + 99991)
    seeds = rng.integers(0, 2**31 - 1, size=K, dtype=np.int64)

    outs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(one_rep_traj)(
            int(sd), X=X, y=y, p=p, Lp=Lp, Tmax=Tmax, ranks=ranks,
            m_top=m_top, T_show=T_show, m_corr=m_corr,
        ) for sd in tqdm(seeds, desc="replicates")
    )

    traj_ad = np.stack([o.traj_ad for o in outs], axis=0)
    traj_vd = np.stack([o.traj_vd for o in outs], axis=0)

    corr_ad_all = {T: np.concatenate([o.corr_ad[T] for o in outs if T in o.corr_ad])
                   for T in T_show}
    corr_vd_all = {T: np.concatenate([o.corr_vd[T] for o in outs if T in o.corr_vd])
                   for T in T_show}

    np.savez_compressed(
        outdir / "raw_data.npz",
        traj_ad=traj_ad, traj_vd=traj_vd,
        ranks=np.array(ranks), Tmax=Tmax,
        n=n, p=p, s=s, snr=snr, L=L, Lp=Lp, K=K, seed=seed,
        **{f"corr_ad_T{T}": corr_ad_all[T] for T in T_show},
        **{f"corr_vd_T{T}": corr_vd_all[T] for T in T_show},
    )
    (outdir / "meta.json").write_text(json.dumps(dict(
        n=n, p=p, s=s, snr=snr, L=L, Lp=Lp, Tmax=Tmax, K=K,
        ranks=ranks, band=list(band), seed=seed, T_show=T_show,
    ), indent=2))

    Tgrid = np.arange(1, Tmax + 1)
    plot_combined(
        Tgrid, traj_ad, traj_vd, ranks,
        corr_ad_all, corr_vd_all,
        T_ecdf=[1, Tmax], T_qq=[1, Tmax],
        outdir=outdir, band=band,
    )
    print("Saved to:", outdir)


if __name__ == "__main__":
    run(
        outdir="results/fig3_distributional_equivalence",
        n=300, p=1000, s=10, snr=1.0, L=5, Tmax=20, K=2000,
        ranks=[1, 5, 20, 50], band=(0.01, 0.99), seed=123, n_jobs=-1,
    )
