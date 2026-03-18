#!/usr/bin/env python
"""
Benchmark worker — runs ONE method on ONE (run, phenotype) pair.
Designed to be spawned as a subprocess for clean time/memory isolation.

Usage:
    python benchmark_worker.py <data_dir> <pheno_dir> <run_id> <method> [snp_subsample]

Outputs a single JSON line to stdout with results.
All logging goes to stderr.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
import warnings

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("VECLIB_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def get_peak_rss_mb():
    try:
        r = resource.getrusage(resource.RUSAGE_SELF)
        return r.ru_maxrss / (1e6 if sys.platform == "darwin" else 1e3)
    except Exception:
        return -1.0


def fdr_tpp(selected, truth):
    sel = set(map(int, np.asarray(selected, dtype=int).ravel()))
    tru = set(map(int, np.asarray(truth, dtype=int).ravel()))
    tp = len(sel & tru)
    fp = len(sel - tru)
    R, P = len(sel), len(tru)
    return (fp / R if R > 0 else 0.0), (tp / P if P > 0 else 0.0)


# =============================================================
# Data loading (mmap/pread preferred, npz fallback)
# =============================================================

def load_data(data_dir, run_id, snp_subsample=None, snp_seed=456):
    """
    Load preprocessed data. Tries in order:
      1. MMapMatrix (pread) — X_std.dat + X_raw.dat, near-zero RSS
      2. np.memmap         — X_raw.dat only, standardize on fly
      3. data.npz          — full load fallback

    Returns (Xc, X_raw, mmap_fd, load_mode) or None.
    mmap_fd >= 0 means pread is available for T-Rex.
    """
    import json as _json

    run_dir = Path(data_dir) / f"run_{run_id}"
    meta_path = run_dir / "meta.json"
    xstd_dat = run_dir / "X_std.dat"
    xraw_dat = run_dir / "X_raw.dat"
    npz_path = run_dir / "data.npz"

    _mm_holders = []  # keep MMapMatrix objects alive
    mmap_fd = -1
    load_mode = "npz"

    # Path 1: MMapMatrix (pread) — zero-copy, minimal RSS
    if xstd_dat.exists() and xraw_dat.exists() and meta_path.exists():
        try:
            from vd_selectors import MMapMatrix
            meta = _json.load(open(meta_path))
            n, p = meta["n_samples"], meta["p_pruned_total"]
            mm_std = MMapMatrix(str(xstd_dat), n, p, writable=False)
            mm_raw = MMapMatrix(str(xraw_dat), n, p, writable=False)
            _mm_holders.extend([mm_std, mm_raw])
            Xc = mm_std.as_array()
            X_raw = mm_raw.as_array()
            mmap_fd = mm_std.fileno()
            load_mode = "pread"
        except Exception as e:
            log(f"MMapMatrix failed: {e}, trying memmap")
            _mm_holders.clear()
            mmap_fd = -1

    # Path 2: memmap X_raw.dat only — standardize on the fly
    if mmap_fd < 0 and xraw_dat.exists() and meta_path.exists():
        try:
            meta = _json.load(open(meta_path))
            n, p = meta["n_samples"], meta["p_pruned_total"]
            X_raw = np.memmap(xraw_dat, dtype="float64", mode="r", shape=(n, p), order="F")
            Xc = np.array(X_raw, dtype=np.float64, order="F")
            Xc -= Xc.mean(axis=0, keepdims=True)
            norms = np.linalg.norm(Xc, axis=0, keepdims=True)
            norms[norms == 0] = 1.0
            Xc /= norms
            load_mode = "memmap"
        except Exception as e:
            log(f"memmap failed: {e}, trying npz")

    # Path 3: npz fallback
    if load_mode == "npz":
        if not npz_path.exists():
            return None
        d = np.load(npz_path)
        Xc = np.asarray(d["X"], dtype=np.float64)
        X_raw = np.asarray(d["X_raw"], dtype=np.float64)
        d.close()

    n, p = Xc.shape

    # SNP subsampling — applies to ALL load modes
    if snp_subsample is not None:
        k = int(snp_subsample)
        p = Xc.shape[1]
        if k < p:
            rng = np.random.default_rng(snp_seed + run_id * 100003)
            idx = np.sort(rng.choice(p, size=k, replace=False))
            Xc = np.asarray(Xc[:, idx], dtype=np.float64).copy()
            X_raw = np.asarray(X_raw[:, idx], dtype=np.float64).copy()
            # pread fd refers to full matrix — invalid after subsample
            mmap_fd = -1
            _mm_holders.clear()
            import gc; gc.collect()

    return Xc, X_raw, mmap_fd, load_mode, _mm_holders


def load_phenotype(pheno_dir, run_id):
    """
    Load pre-generated phenotype.
    If the pheno file contains subsampled Xc/X_raw (from Phase 0),
    return those too so the worker doesn't need to load from data_dir.

    Returns (y, causal_idx, Xc_or_None, X_raw_or_None).
    """
    path = Path(pheno_dir) / f"pheno_run_{run_id}.npz"
    if not path.exists():
        return None
    d = np.load(path)
    y = np.asarray(d["y"], dtype=np.float64)
    causal_idx = np.asarray(d["causal_idx"], dtype=int)

    Xc = None
    X_raw = None
    if "Xc" in d and "X_raw" in d:
        Xc = np.asarray(d["Xc"], dtype=np.float64)
        X_raw = np.asarray(d["X_raw"], dtype=np.float64)

    d.close()
    return y, causal_idx, Xc, X_raw


# =============================================================
# Method implementations
# =============================================================

def run_trex(Xc, y, cfg, mmap_fd=-1):
    from vd_selectors import (
        TRexSelector, TRexOptions, SolverType, CalibMode, VDDummyLaw,
    )
    opt = TRexOptions()
    opt.tFDR = float(cfg["alpha"])
    opt.K = cfg.get("K", 20)
    opt.L_factor = cfg.get("L_factor", 5)
    opt.T_stop = cfg.get("T_stop", -1)
    opt.calib = getattr(CalibMode, cfg.get("calib", "CalibrateT"))
    opt.solver = getattr(SolverType, cfg.get("solver", "LARS"))
    opt.dummy_law = VDDummyLaw.Spherical
    opt.posthoc_mode = cfg.get("posthoc_mode", False)
    opt.stride_width = cfg.get("stride_width", 5)
    opt.max_stale_strides = cfg.get("max_stale_strides", 2)
    opt.n_threads = cfg.get("n_threads", 1)
    opt.verbose = False
    opt.seed = cfg.get("seed", 42)
    if "rho" in cfg:
        opt.rho = cfg["rho"]

    # Enable pread if mmap is available
    if mmap_fd >= 0:
        opt.mmap_fd = mmap_fd
        opt.mmap_block_cols = 512

    selector = TRexSelector(opt)

    # When using mmap, Xc is already F-order — no copy needed
    if mmap_fd >= 0:
        Xf = Xc
    else:
        Xf = np.asfortranarray(Xc, dtype=np.float64)

    res = selector.run(Xf, y)
    sel = np.asarray(res.selected_var, dtype=int)
    extra = {
        "T_stop": int(getattr(res, "T_stop", 0)),
        "L": int(getattr(res, "num_dummies", 0)),
        "L_calibrated": int(getattr(res, "L_calibrated", 0)),
    }
    return sel, extra


def run_bh(X_raw, y, cfg):
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    pvals, test_mode = _compute_pvalues(X_raw, y, cfg)
    reject, _, _, _ = multipletests(pvals, alpha=cfg["alpha"], method="fdr_bh")
    return np.where(reject)[0].astype(int), {"test_mode": test_mode}


def run_by(X_raw, y, cfg):
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    pvals, test_mode = _compute_pvalues(X_raw, y, cfg)
    reject, _, _, _ = multipletests(pvals, alpha=cfg["alpha"], method="fdr_by")
    return np.where(reject)[0].astype(int), {"test_mode": test_mode}


def _compute_pvalues(X_raw, y, cfg):
    from scipy import stats

    n, p = X_raw.shape
    pheno_model = cfg.get("pheno_model", "multiplicative_rr")

    if n > 2 * p:
        try:
            pvals = _ols_pvalues(X_raw, y)
            if np.all(np.isfinite(pvals)) and np.all(pvals >= 0) and np.all(pvals <= 1):
                return pvals, "OLS"
        except Exception:
            pass

    if pheno_model == "linear":
        return _marginal_regression_pvalues(X_raw, y), "marginal_reg"
    else:
        y_binary = (y > 0).astype(np.float64)
        return _cochran_armitage_pvalues(X_raw, y_binary), "marginal_CA"


def _ols_pvalues(X, y):
    from scipy import stats
    n, p = X.shape
    X1 = np.column_stack([np.ones(n), X])
    beta, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    dof = n - p - 1
    sigma2 = np.sum(resid ** 2) / max(dof, 1)
    try:
        XtX_inv = np.linalg.inv(X1.T @ X1)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X1.T @ X1)
    se = np.sqrt(np.maximum(sigma2 * np.diag(XtX_inv)[1:], 1e-30))
    t_stat = beta[1:] / se
    return 2.0 * stats.t.sf(np.abs(t_stat), df=max(dof, 1))


def _marginal_regression_pvalues(X_raw, y):
    from scipy import stats
    n = len(y)
    y_c = y - y.mean()
    X_c = X_raw - X_raw.mean(axis=0, keepdims=True)
    norms_x = np.linalg.norm(X_c, axis=0)
    norms_x = np.maximum(norms_x, 1e-20)
    r = (X_c.T @ y_c) / (norms_x * np.linalg.norm(y_c))
    r = np.clip(r, -1 + 1e-10, 1 - 1e-10)
    t_stat = r * np.sqrt((n - 2) / (1 - r ** 2))
    return 2.0 * stats.t.sf(np.abs(t_stat), df=n - 2)


def _cochran_armitage_pvalues(X_raw, y_binary):
    from scipy import stats
    n = len(y_binary)
    n1 = y_binary.sum()
    n0 = n - n1
    col_sums = X_raw.sum(axis=0)
    col_sq_sums = (X_raw ** 2).sum(axis=0)
    T_obs = y_binary @ X_raw
    T_exp = (n1 / n) * col_sums
    var_T = (n0 * n1) / (n * (n - 1)) * (n * col_sq_sums - col_sums ** 2)
    var_T = np.maximum(var_T, 1e-20)
    Z = (T_obs - T_exp) / np.sqrt(var_T)
    return 2.0 * stats.norm.sf(np.abs(Z))


def run_knockoff_identity(Xc, y, cfg):
    rng = np.random.default_rng(cfg.get("seed", 42))
    n, p = Xc.shape
    Xk = rng.standard_normal((n, p))
    z = np.abs(Xc.T @ y)
    zk = np.abs(Xk.T @ y)
    W = z - zk
    tau = _knockoff_threshold_plus(W, cfg["alpha"])
    if not np.isfinite(tau):
        return np.array([], dtype=int), {}
    return np.where(W >= tau)[0].astype(int), {}


def run_knockoff_knockpy(Xc, y, cfg):
    import knockpy
    n, p = Xc.shape

    if n >= 2 * p:
        ksampler = "fx"
        ko_type = "fixed-X"
    else:
        ksampler = "gaussian"
        ko_type = "model-X_Gaussian"

    # knockpy auto-detects binary y and switches to logistic regression,
    # which fails on centered binary phenotypes (values like -0.3, 0.7).
    # Adding tiny noise ensures y is treated as continuous (lasso regression).
    y_ko = y.copy()
    if len(np.unique(y_ko)) <= 2:
        y_ko = y_ko + np.random.RandomState(0).randn(n) * 1e-6

    kfilter = knockpy.KnockoffFilter(
        ksampler=ksampler,
        fstat="lasso",
    )

    # knockpy prints progress to stdout, which corrupts our JSON output.
    # Redirect stdout → stderr during the call.
    import io
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        rej = kfilter.forward(X=Xc, y=y_ko, fdr=cfg["alpha"])
    finally:
        sys.stdout = old_stdout

    sel = np.where(rej)[0].astype(int)
    return sel, {"ko_type": ko_type}


def _knockoff_threshold_plus(W, q):
    W = np.asarray(W, dtype=float).ravel()
    ts = np.sort(np.unique(np.abs(W[W != 0])))
    if ts.size == 0:
        return np.inf
    for t in ts:
        num = 1 + np.sum(W <= -t)
        den = max(1, np.sum(W >= t))
        if num / den <= q:
            return float(t)
    return np.inf


# =============================================================
# Method dispatch
# =============================================================

METHODS = {
    "trex":        lambda Xc, X_raw, y, cfg, mmap_fd: run_trex(Xc, y, cfg, mmap_fd),
    "bh":          lambda Xc, X_raw, y, cfg, mmap_fd: run_bh(X_raw, y, cfg),
    "by":          lambda Xc, X_raw, y, cfg, mmap_fd: run_by(X_raw, y, cfg),
    "ko_id":       lambda Xc, X_raw, y, cfg, mmap_fd: run_knockoff_identity(Xc, y, cfg),
    "ko_knockpy":  lambda Xc, X_raw, y, cfg, mmap_fd: run_knockoff_knockpy(Xc, y, cfg),
}


# =============================================================
# Main
# =============================================================

def main():
    if len(sys.argv) < 5:
        log(f"Usage: {sys.argv[0]} <data_dir> <pheno_dir> <run_id> <method> [snp_subsample] [cfg_json]")
        sys.exit(1)

    data_dir = sys.argv[1]
    pheno_dir = sys.argv[2]
    run_id = int(sys.argv[3])
    method = sys.argv[4]
    snp_subsample = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "" else None
    cfg_json = sys.argv[6] if len(sys.argv) > 6 else "{}"

    cfg = json.loads(cfg_json)
    cfg.setdefault("alpha", 0.1)
    cfg.setdefault("pheno_model", "multiplicative_rr")

    if method not in METHODS:
        log(f"Unknown method: {method}. Available: {list(METHODS.keys())}")
        sys.exit(1)

    # Load phenotype first — may contain bundled data from Phase 0
    pheno = load_phenotype(pheno_dir, run_id)
    if pheno is None:
        json.dump({"run": run_id, "method": method, "error": "phenotype not found"}, sys.stdout)
        sys.exit(0)
    y, causal_idx, Xc_bundled, X_raw_bundled = pheno

    # Use bundled data if available (subsampled in Phase 0),
    # otherwise load from data_dir (full p, pread for T-Rex)
    mmap_fd = -1
    _mm_holders = []
    if Xc_bundled is not None:
        Xc = Xc_bundled
        X_raw = X_raw_bundled
        load_mode = "bundled"
    else:
        loaded = load_data(data_dir, run_id, snp_subsample=None)  # no subsample — use full p
        if loaded is None:
            json.dump({"run": run_id, "method": method, "error": "data not found"}, sys.stdout)
            sys.exit(0)
        Xc, X_raw, mmap_fd, load_mode, _mm_holders = loaded

    n, p = Xc.shape
    log(f"run={run_id} method={method} n={n} p={p} s={len(causal_idx)} load={load_mode}")

    # Run method
    t0 = time.perf_counter()
    try:
        selected, extra = METHODS[method](Xc, X_raw, y, cfg, mmap_fd)
    except Exception as e:
        dt = time.perf_counter() - t0
        result = {
            "run": run_id, "method": method,
            "error": str(e), "runtime_s": dt,
            "peak_rss_mb": get_peak_rss_mb(),
            "n": n, "p": p, "load_mode": load_mode,
        }
        json.dump(result, sys.stdout)
        sys.exit(0)

    dt = time.perf_counter() - t0
    fdp, tpp = fdr_tpp(selected, causal_idx)

    result = {
        "run": run_id,
        "method": method,
        "fdp": fdp,
        "tpp": tpp,
        "n_disc": len(selected),
        "runtime_s": round(dt, 4),
        "peak_rss_mb": round(get_peak_rss_mb(), 1),
        "n": n,
        "p": p,
        "n_causal": len(causal_idx),
        "load_mode": load_mode,
    }
    result.update(extra)

    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
