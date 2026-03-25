from __future__ import annotations

import os
import json
import time
import platform
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest
from joblib import Parallel, delayed
from tqdm import tqdm

from AD_LARS import AD_LARS

EPS = 1e-12


# ----------------------------
# utils
# ----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)


# ----------------------------
# standardization
# ----------------------------
def center_unitvar(M: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Column-center, then scale each column to unit sample std (ddof=1)."""
    M = np.asarray(M, float)
    M -= M.mean(axis=0, keepdims=True)
    s = M.std(axis=0, ddof=1, keepdims=True)
    s = np.clip(s, eps, None)
    M /= s
    return np.asfortranarray(M)


def center(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    return y - y.mean()


# ----------------------------
# dummy generation
# ----------------------------
def make_dummies(
    dist: str,
    n: int,
    Lp: int,
    rng: np.random.Generator,
    *,
    pareto_alpha: float = 4.0,
    t_df: float = 3.0,
    exp_scale: float = 1.0,
    lognorm_sigma: float = 1.0,
) -> np.ndarray:
    dist = dist.lower()

    if dist in {"gaussian", "normal"}:
        D = rng.standard_normal((n, Lp))

    elif dist in {"rademacher", "rad"}:
        D = rng.choice([-1.0, 1.0], size=(n, Lp))

    elif dist in {"pareto"}:
        # heavy-tailed, positive skew
        D = rng.pareto(float(pareto_alpha), size=(n, Lp)) + 1.0

    elif dist in {"t", "student"}:
        D = rng.standard_t(float(t_df), size=(n, Lp))

    elif dist in {"exponential", "exp"}:
        scale = float(exp_scale)
        if scale <= 0:
            raise ValueError("exp_scale must be > 0")
        D = rng.exponential(scale=scale, size=(n, Lp))
        # mean=scale, var=scale^2 -> standardize
        D = (D - scale) / scale

    elif dist in {"lognormal", "lognorm"}:
        sigma = float(lognorm_sigma)
        if sigma <= 0:
            raise ValueError("lognorm_sigma must be > 0")
        D = rng.lognormal(mean=0.0, sigma=sigma, size=(n, Lp))
        mean = float(np.exp(sigma**2 / 2.0))
        var = float((np.exp(sigma**2) - 1.0) * np.exp(sigma**2))
        D = (D - mean) / np.sqrt(var)

    else:
        raise ValueError(f"Unknown dist={dist}")

    return center_unitvar(D)


# ----------------------------
# metrics
# ----------------------------
def w1_to_standard_normal(x: np.ndarray) -> float:
    """
    1D Wasserstein-1 distance between empirical sample x and N(0,1),
    computed via quantile coupling (deterministic, no Monte Carlo).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    m = x.size
    if m == 0:
        return float("nan")
    xs = np.sort(x)
    # mid-point plotting positions
    u = (np.arange(m) + 0.5) / m
    z = norm.ppf(u)
    return float(np.mean(np.abs(xs - z)))


# ----------------------------
# one replicate
# ----------------------------
def one_rep_metrics(
    rep_seed: int,
    X: np.ndarray,
    y: np.ndarray,
    p_real: int,
    Lp: int,
    K_steps: int,
    dist: str,
    pareto_alpha: float,
    t_df: float,
    exp_scale: float,
    lognorm_sigma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(rep_seed)
    D = make_dummies(
        dist,
        X.shape[0],
        Lp,
        rng,
        pareto_alpha=pareto_alpha,
        t_df=t_df,
        exp_scale=exp_scale,
        lognorm_sigma=lognorm_sigma,
    )
    XD = np.hstack([X, D])

    tl = AD_LARS(XD, y, num_dummies=Lp, normalize=False, eps=EPS, verbose=False, track_basis=True)
    tl.run(T=K_steps, stop="steps")

    dummy_entry = tl.entry_step[p_real:]  # length Lp

    KS = np.full(K_steps, np.nan, dtype=float)
    W1 = np.full(K_steps, np.nan, dtype=float)
    MAX_W = np.full(K_steps, np.nan, dtype=float)

    for k in range(1, K_steps + 1):
        if tl.basis.shape[1] <= k:
            continue
        e_k = tl.basis[:, k]
        MAX_W[k - 1] = float(np.max(np.abs(e_k)))

        # dummies unused up to step k-1
        unused = np.where((dummy_entry < 0) | (dummy_entry > (k - 1)))[0]
        if unused.size < 20:
            continue

        x = (D[:, unused].T @ e_k)
        x = x[np.isfinite(x)]
        if x.size < 20:
            continue

        KS[k - 1] = float(kstest(x, cdf=norm.cdf).statistic)
        W1[k - 1] = w1_to_standard_normal(x)

    return KS, W1, MAX_W


# ----------------------------
# experiment config
# ----------------------------
@dataclass
class Config:
    ns: List[int]
    p_real: int = 500
    L: int = 5
    K_steps: int = 10
    R: int = 1000

    pareto_alpha: float = 4.0
    t_df: float = 3.0
    exp_scale: float = 1.0
    lognorm_sigma: float = 1.0

    dists: Tuple[str, ...] = ("gaussian", "exponential", "pareto", "rademacher", "t")

    seed: int = 42
    ks_to_plot: Tuple[int, ...] = (1, 5, 10)
    n_jobs: int = -1
    show_inner_tqdm: bool = True

    # output
    results_root: str = "results"
    run_name: str = "lemma6_multiK"


# ----------------------------
# runner
# ----------------------------
def run_experiment(cfg: Config) -> str:
    run_id = f"{cfg.run_name}"
    out_dir = os.path.join(cfg.results_root, run_id)
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "arrays"))

    # Save config + meta
    write_json(os.path.join(out_dir, "config.json"), asdict(cfg))
    meta = {
        "run_id": run_id,
        "start_time_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }
    write_json(os.path.join(out_dir, "meta.json"), meta)

    rng0 = np.random.default_rng(cfg.seed)

    # dict[dist][k] -> list over n
    ks_med = {dist: {k: [] for k in cfg.ks_to_plot} for dist in cfg.dists}
    w1_med = {dist: {k: [] for k in cfg.ks_to_plot} for dist in cfg.dists}
    mw_med = {dist: {k: [] for k in cfg.ks_to_plot} for dist in cfg.dists}

    total_blocks = len(cfg.ns) * len(cfg.dists)
    outer_pbar = tqdm(total=total_blocks, desc="(n, dist) blocks", position=0)

    for n in cfg.ns:
        print(f"\n--- Testing n={n} ---")
        rng_env = np.random.default_rng(n + cfg.seed)
        X = center_unitvar(rng_env.standard_normal((n, cfg.p_real)))
        y = center(rng_env.standard_normal(n))

        Lp = cfg.L * cfg.p_real
        rep_seeds = rng0.integers(0, 2**31 - 1, size=cfg.R, dtype=np.int64)

        for dist in cfg.dists:
            outer_pbar.set_postfix_str(f"n={n}, dist={dist}")

            rep_iter = tqdm(
                rep_seeds,
                desc=f"n={n} {dist}",
                leave=False,
                position=1,
                disable=not cfg.show_inner_tqdm,
            )

            outs = Parallel(n_jobs=cfg.n_jobs)(
                delayed(one_rep_metrics)(
                    int(sd),
                    X,
                    y,
                    cfg.p_real,
                    Lp,
                    cfg.K_steps,
                    dist,
                    cfg.pareto_alpha,
                    cfg.t_df,
                    cfg.exp_scale,
                    cfg.lognorm_sigma,
                )
                for sd in rep_iter
            )

            ks_vals = np.stack([o[0] for o in outs], axis=0)  # (R, K_steps)
            w1_vals = np.stack([o[1] for o in outs], axis=0)  # (R, K_steps)
            mw_vals = np.stack([o[2] for o in outs], axis=0)  # (R, K_steps)

            # Save all arrays for this (n, dist)
            arr_path = os.path.join(out_dir, "arrays", f"n{n}_{dist}.npz")
            np.savez_compressed(
                arr_path,
                ks=ks_vals.astype(np.float32),
                w1=w1_vals.astype(np.float32),
                maxw=mw_vals.astype(np.float32),
                n=np.int32(n),
                dist=dist,
                p_real=np.int32(cfg.p_real),
                Lp=np.int32(Lp),
                K_steps=np.int32(cfg.K_steps),
                R=np.int32(cfg.R),
            )

            # medians for plotting
            for k in cfg.ks_to_plot:
                idx = k - 1
                ks_med[dist][k].append(float(np.nanmedian(ks_vals[:, idx])))
                w1_med[dist][k].append(float(np.nanmedian(w1_vals[:, idx])))
                mw_med[dist][k].append(float(np.nanmedian(mw_vals[:, idx])))

            outer_pbar.update(1)

    outer_pbar.close()

    # Save summaries as JSON (small, human-readable)
    summary = {
        "ns": cfg.ns,
        "ks_to_plot": list(cfg.ks_to_plot),
        "ks_median": ks_med,
        "w1_median": w1_med,
        "maxw_median": mw_med,
    }
    write_json(os.path.join(out_dir, "summary.json"), summary)

    # ---------- Plot KS: 1xlen(ks_to_plot) panels ----------
    fig, axes = plt.subplots(
        1, len(cfg.ks_to_plot),
        figsize=(5.8 * len(cfg.ks_to_plot), 4.8),
        sharey=True
    )
    if len(cfg.ks_to_plot) == 1:
        axes = [axes]

    for ax, k in zip(axes, cfg.ks_to_plot):
        for dist in cfg.dists:
            ax.loglog(cfg.ns, ks_med[dist][k], marker="o", label=dist)
        ax.set_title(f"KS vs n (step k={k})")
        ax.set_xlabel("n")
        ax.grid(True, which="both", ls="-", alpha=0.35)
    axes[0].set_ylabel("Median KS to N(0,1)")
    axes[-1].legend()
    fig.suptitle("Lemma 6 diagnostic: KS over n at multiple steps")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lemma6_ks_convergence_multiK.pdf"), dpi=220)
    fig.savefig(os.path.join(out_dir, "lemma6_ks_convergence_multiK.png"), dpi=220)
    plt.close(fig)

    # ---------- Plot W1: 1xlen(ks_to_plot) panels ----------
    fig, axes = plt.subplots(
        1, len(cfg.ks_to_plot),
        figsize=(5.8 * len(cfg.ks_to_plot), 4.8),
        sharey=True
    )
    if len(cfg.ks_to_plot) == 1:
        axes = [axes]

    for ax, k in zip(axes, cfg.ks_to_plot):
        for dist in cfg.dists:
            ax.loglog(cfg.ns, w1_med[dist][k], marker="o", label=dist)
        ax.set_title(f"W1 vs n (step k={k})")
        ax.set_xlabel("n")
        ax.grid(True, which="both", ls="-", alpha=0.35)
    axes[0].set_ylabel("Median W1 to N(0,1)")
    axes[-1].legend()
    fig.suptitle("Lemma 6 diagnostic: W1 over n at multiple steps")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lemma6_w1_convergence_multiK.pdf"), dpi=220)
    fig.savefig(os.path.join(out_dir, "lemma6_w1_convergence_multiK.png"), dpi=220)
    plt.close(fig)

    # ---------- Plot delocalization: 1xlen(ks_to_plot) panels ----------
    fig, axes = plt.subplots(
        1, len(cfg.ks_to_plot),
        figsize=(5.8 * len(cfg.ks_to_plot), 4.8),
        sharey=True
    )
    if len(cfg.ks_to_plot) == 1:
        axes = [axes]

    for ax, k in zip(axes, cfg.ks_to_plot):
        for dist in cfg.dists:
            ax.loglog(cfg.ns, mw_med[dist][k], marker="s", ls="--", label=dist)
        ax.set_title(f"max|e_k| vs n (step k={k})")
        ax.set_xlabel("n")
        ax.grid(True, which="both", ls="-", alpha=0.35)
    axes[0].set_ylabel("Median max(|e_k|)")
    axes[-1].legend()
    fig.suptitle("Assumption D diagnostic: delocalization over n at multiple steps")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lemma6_delocalization_multiK.pdf"), dpi=220)
    fig.savefig(os.path.join(out_dir, "lemma6_delocalization_multiK.png"), dpi=220)
    plt.close(fig)

    # finalize meta
    meta["end_time_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(os.path.join(out_dir, "meta.json"), meta)

    print(f"\nSaved everything to: {out_dir}")
    return out_dir


if __name__ == "__main__":
    config = Config(
        ns=[50, 100, 250, 500, 1000, 5000, 100000],
        p_real=100,
        L = 10,
        K_steps=10,
        R=5000,
        pareto_alpha=3.0,
        t_df=3.0,
        exp_scale=1.0,
        lognorm_sigma=1.0,
        dists=("gaussian", "exponential", "pareto", "rademacher","lognormal", "t"),
        seed=42,
        ks_to_plot=(1, 5, 10),
        n_jobs=-1,
        show_inner_tqdm=True,
        results_root="results",
        run_name="fig5_universality_diagnostics",
    )
    run_experiment(config)