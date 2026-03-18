#!/usr/bin/env python
"""
Generate phenotype for a single run.

Usage:
    python generate_phenotype.py <data_dir> <pheno_dir> <run_id> <pheno_model> <cfg_json>

Output:
    <pheno_dir>/pheno_run_<run_id>.npz
    Contains: y, causal_idx, seed, n, p
    If snp_subsample is set: also Xc, X_raw (bundled for worker)
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

warnings.filterwarnings("ignore")


def generate_phenotype(
    Xc, X_raw, causal_idx, pheno_model, h2, prevalence, rng,
    het_rr_range=(1.2, 1.8),
):
    n, p = Xc.shape
    s = len(causal_idx)
    beta = np.zeros(p)
    beta[causal_idx] = rng.choice([-1.0, 1.0], size=s)
    signal = Xc @ beta
    sig_var = max(float(np.var(signal)), 1e-20)

    if h2 >= 1.0:
        noise = np.zeros(n)
    else:
        noise_var = sig_var * (1.0 - h2) / h2
        noise = np.sqrt(noise_var) * rng.standard_normal(n)

    liability = signal + noise

    if pheno_model == "linear":
        y = liability.copy()
    elif pheno_model == "liability_binary":
        threshold = np.percentile(liability, 100.0 * (1.0 - prevalence))
        y = (liability >= threshold).astype(np.float64)
    elif pheno_model == "multiplicative_rr":
        risk_allele = rng.integers(0, 2, size=s)
        het_rr = rng.uniform(het_rr_range[0], het_rr_range[1], size=s)
        hom_rr = het_rr ** 2
        log_risk = np.zeros(n)
        for j, ra, hr, hmr in zip(causal_idx, risk_allele, het_rr, hom_rr):
            g = np.round(X_raw[:, j]).astype(int).clip(0, 2)
            if ra == 0:
                g = 2 - g
            rr = np.ones(n)
            rr[g == 1] = hr
            rr[g == 2] = hmr
            log_risk += np.log(rr)
        prob = np.exp(log_risk)
        prob /= prob.sum()
        n_cases = int(n * prevalence + 0.5)
        case_idx = rng.choice(n, size=n_cases, replace=False, p=prob)
        y = np.zeros(n)
        y[case_idx] = 1.0
    else:
        raise ValueError(f"Unknown pheno_model: {pheno_model}")

    return y - y.mean()


def load_data(data_dir, run_id, snp_subsample=None, snp_seed=456):
    """Load data. Tries mmap dat files first, then npz."""
    run_dir = Path(data_dir) / f"run_{run_id}"
    meta_path = run_dir / "meta.json"
    xstd_dat = run_dir / "X_std.dat"
    xraw_dat = run_dir / "X_raw.dat"
    npz_path = run_dir / "data.npz"

    if xstd_dat.exists() and xraw_dat.exists() and meta_path.exists():
        meta = json.load(open(meta_path))
        n, p = meta["n_samples"], meta["p_pruned_total"]
        Xc = np.memmap(xstd_dat, dtype="float64", mode="r", shape=(n, p), order="F")
        X_raw = np.memmap(xraw_dat, dtype="float64", mode="r", shape=(n, p), order="F")
    elif npz_path.exists():
        d = np.load(npz_path)
        Xc = np.asarray(d["X"], dtype=np.float64)
        X_raw = np.asarray(d["X_raw"], dtype=np.float64)
        d.close()
        n, p = Xc.shape
    else:
        return None

    if snp_subsample is not None:
        k = int(snp_subsample)
        if k < p:
            rng = np.random.default_rng(snp_seed + run_id * 100003)
            idx = np.sort(rng.choice(p, size=k, replace=False))
            Xc = np.asarray(Xc[:, idx], dtype=np.float64)
            X_raw = np.asarray(X_raw[:, idx], dtype=np.float64)
            p = k

    return Xc, X_raw, n, p


def main():
    data_dir = sys.argv[1]
    pheno_dir = Path(sys.argv[2])
    run_id = int(sys.argv[3])
    pheno_model = sys.argv[4]
    cfg = json.loads(sys.argv[5])

    cfg.setdefault("s", 10)
    cfg.setdefault("h2", 0.3)
    cfg.setdefault("prevalence", 0.5)
    cfg.setdefault("het_rr_range", [1.05, 1.25])
    cfg.setdefault("seed0", 42)

    pheno_dir.mkdir(parents=True, exist_ok=True)
    pheno_path = pheno_dir / f"pheno_run_{run_id}.npz"

    if pheno_path.exists():
        print(f"run {run_id}: already exists, skipping")
        return

    snp_sub = cfg.get("snp_subsample")
    snp_seed = cfg.get("snp_seed", 456)

    loaded = load_data(data_dir, run_id, snp_sub, snp_seed)
    if loaded is None:
        print(f"run {run_id}: data not found")
        sys.exit(1)

    Xc, X_raw, n, p = loaded

    seed = cfg["seed0"] + run_id * 10_007
    rng = np.random.default_rng(seed)
    causal_idx = np.sort(rng.choice(p, size=cfg["s"], replace=False)).astype(int)

    y = generate_phenotype(
        Xc, X_raw, causal_idx,
        pheno_model=pheno_model,
        h2=cfg["h2"],
        prevalence=cfg["prevalence"],
        rng=rng,
        het_rr_range=tuple(cfg["het_rr_range"]),
    )

    save_dict = dict(y=y, causal_idx=causal_idx, seed=seed, n=n, p=p)
    # Bundle subsampled data so workers don't re-load the full matrix
    if snp_sub is not None:
        save_dict["Xc"] = Xc
        save_dict["X_raw"] = X_raw

    np.savez_compressed(pheno_path, **save_dict)
    print(f"run {run_id}: n={n} p={p} s={len(causal_idx)} "
          f"model={pheno_model} h2={cfg['h2']} prev={cfg['prevalence']}")


if __name__ == "__main__":
    main()
