#!/usr/bin/env python
"""
HAPNEST Benchmark Orchestrator
================================
Phase 0: Generate and save phenotypes for all runs (deterministic, reusable)
Phase 1: For each (run, method), spawn a child process via benchmark_worker.py
          → clean time/memory isolation per method
Phase 2: Collect results, print summary, save CSVs

Handles SIGTERM gracefully (saves partial results before exit).
Designed for SLURM 1-day partition.

Usage:
    python hapnest_benchmark.py <data_dir> <n_runs> <pheno_model> [run_subsample] [snp_subsample]

Examples:
    python hapnest_benchmark.py data_preprocessed 30 multiplicative_rr
    python hapnest_benchmark.py data_preprocessed 30 linear 10 2000
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("VECLIB_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

WORKER_SCRIPT = str(Path(__file__).parent / "benchmark_worker.py")


# =============================================================
# Phenotype generation
# =============================================================

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


def load_run_data(base_dir, run_id, snp_subsample=None, snp_seed=456):
    """Load data for phenotype generation. Supports mmap and npz paths."""
    import json as _json

    run_dir = Path(base_dir) / f"run_{run_id}"
    meta_path = run_dir / "meta.json"
    xstd_dat = run_dir / "X_std.dat"
    xraw_dat = run_dir / "X_raw.dat"
    npz_path = run_dir / "data.npz"

    Xc = X_raw = None

    # Path 1: mmap dat files
    if xstd_dat.exists() and xraw_dat.exists() and meta_path.exists():
        meta = _json.load(open(meta_path))
        n, p = meta["n_samples"], meta["p_pruned_total"]
        X_raw = np.memmap(xraw_dat, dtype="float64", mode="r", shape=(n, p), order="F")
        Xc = np.memmap(xstd_dat, dtype="float64", mode="r", shape=(n, p), order="F")
    elif xraw_dat.exists() and meta_path.exists():
        meta = _json.load(open(meta_path))
        n, p = meta["n_samples"], meta["p_pruned_total"]
        X_raw = np.memmap(xraw_dat, dtype="float64", mode="r", shape=(n, p), order="F")
        Xc = np.array(X_raw, dtype=np.float64, order="F")
        Xc -= Xc.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(Xc, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        Xc /= norms
    elif npz_path.exists():
        d = np.load(npz_path)
        Xc = np.asarray(d["X"], dtype=np.float64)
        X_raw = np.asarray(d["X_raw"], dtype=np.float64)
        d.close()
    else:
        return None

    if snp_subsample is not None:
        p = Xc.shape[1]
        k = int(snp_subsample)
        if k < p:
            rng = np.random.default_rng(snp_seed + run_id * 100003)
            idx = np.sort(rng.choice(p, size=k, replace=False))
            Xc = np.asarray(Xc[:, idx], dtype=np.float64)
            X_raw = np.asarray(X_raw[:, idx], dtype=np.float64)

    return Xc, X_raw


def generate_all_phenotypes(data_dir, run_ids, cfg, pheno_dir, snp_subsample=None, snp_seed=456):
    pheno_dir = Path(pheno_dir)
    pheno_dir.mkdir(parents=True, exist_ok=True)

    generated = skipped = failed = 0

    for rid in tqdm(run_ids, desc="Generating phenotypes"):
        pheno_path = pheno_dir / f"pheno_run_{rid}.npz"
        if pheno_path.exists():
            skipped += 1
            continue

        loaded = load_run_data(data_dir, rid, snp_subsample, snp_seed)
        if loaded is None:
            failed += 1
            continue

        Xc, X_raw = loaded
        n, p = Xc.shape
        seed = cfg["seed0"] + rid * 10_007
        rng = np.random.default_rng(seed)
        causal_idx = np.sort(rng.choice(p, size=cfg["s"], replace=False)).astype(int)

        y = generate_phenotype(
            Xc, X_raw, causal_idx,
            pheno_model=cfg["pheno_model"],
            h2=cfg["h2"],
            prevalence=cfg["prevalence"],
            rng=rng,
            het_rr_range=cfg["het_rr_range"],
        )

        # Save phenotype + data. When subsampled, include Xc/X_raw so
        # workers don't need to reload and re-subsample the full matrix.
        save_dict = dict(y=y, causal_idx=causal_idx, seed=seed, n=n, p=p)
        if snp_subsample is not None:
            save_dict["Xc"] = np.asarray(Xc, dtype=np.float64)
            save_dict["X_raw"] = np.asarray(X_raw, dtype=np.float64)
            save_dict["subsampled"] = True
        else:
            save_dict["subsampled"] = False

        np.savez_compressed(pheno_path, **save_dict)
        generated += 1
        del Xc, X_raw, y

    print(f"  Phenotypes: {generated} generated, {skipped} cached, {failed} failed\n")
    return generated + skipped


# =============================================================
# Worker spawning
# =============================================================

def run_worker(data_dir, pheno_dir, run_id, method, snp_subsample, cfg, timeout=None):
    cmd = [
        sys.executable, WORKER_SCRIPT,
        str(data_dir), str(pheno_dir), str(run_id), method,
        str(snp_subsample) if snp_subsample else "",
        json.dumps(cfg),
    ]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, env={**os.environ},
        )
        if proc.stdout.strip():
            result = json.loads(proc.stdout.strip())
        else:
            result = {"run": run_id, "method": method,
                      "error": f"no output; stderr: {proc.stderr[-200:]}"}
        if proc.returncode != 0 and "error" not in result:
            result["error"] = f"exit code {proc.returncode}; {proc.stderr[-200:]}"

    except subprocess.TimeoutExpired:
        result = {"run": run_id, "method": method,
                  "error": f"timeout ({timeout}s)", "runtime_s": timeout}
    except Exception as e:
        result = {"run": run_id, "method": method, "error": str(e)}

    return result


# =============================================================
# Result store with incremental save + signal handling
# =============================================================

class ResultStore:
    def __init__(self, outdir: Path):
        self.outdir = outdir
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self._killed = False
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print(f"\n*** Signal {signum} — saving partial results ***", flush=True)
        self._killed = True
        self.flush()
        sys.exit(0)

    @property
    def killed(self):
        return self._killed

    def add(self, result: dict):
        self.rows.append(result)

    def flush(self):
        if not self.rows:
            return
        df = pd.DataFrame(self.rows)
        df.to_csv(self.outdir / "all_results.csv", index=False)
        self._save_pivoted(df)

    def _save_pivoted(self, df):
        rows = []
        for rid, grp in df.groupby("run"):
            row = {"run": rid}
            for _, r in grp.iterrows():
                m = r["method"]
                for col in ["fdp", "tpp", "n_disc", "runtime_s", "peak_rss_mb"]:
                    if col in r and pd.notna(r[col]):
                        row[f"{m}_{col}"] = r[col]
                if "error" in r and pd.notna(r.get("error")):
                    row[f"{m}_error"] = r["error"]
                for col in ["T_stop", "L", "L_calibrated", "test_mode", "ko_type"]:
                    if col in r and pd.notna(r.get(col)):
                        row[f"{m}_{col}"] = r[col]
            rows.append(row)
        pd.DataFrame(rows).to_csv(self.outdir / "per_run.csv", index=False)

    def save_meta(self, meta: dict):
        with open(self.outdir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)


# =============================================================
# Subsample helpers
# =============================================================

def parse_subsample_arg(arg, n_available):
    if arg is None:
        return None, None
    txt = str(arg).strip()
    if not txt:
        return None, None
    if any(ch in txt for ch in [".", "e", "E"]):
        frac = float(txt)
        k = max(1, int(np.floor(frac * n_available)))
        return min(k, n_available), frac
    k = int(txt)
    return min(max(k, 1), n_available), None


def choose_run_subset(run_ids, subsample_arg, seed):
    n = len(run_ids)
    k, frac = parse_subsample_arg(subsample_arg, n)
    if k is None:
        return list(run_ids), {"mode": "all", "selected": n}
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(np.asarray(run_ids), size=k, replace=False).tolist())
    return chosen, {"mode": "subsample", "requested": frac or k, "selected": len(chosen), "seed": seed}


# =============================================================
# Summary
# =============================================================

def print_summary(df, methods, alpha):
    print("\n" + "=" * 120)
    print(f"SUMMARY ({len(df)} results across {df['run'].nunique()} runs)")
    print("=" * 120)
    print(
        f"{'Method':<35} {'Runs':>5} {'Mean FDP':>10} {'Std FDP':>9} "
        f"{'Mean TPP':>10} {'Std TPP':>9} {'Mean disc':>10} "
        f"{'Mean t(s)':>10} {'Mean RSS':>10} {'FDR ctrl':>9}"
    )
    print("-" * 120)

    detail_rows = []
    for method in methods:
        mdf = df[(df["method"] == method) & df["fdp"].notna()] if "fdp" in df.columns else pd.DataFrame()
        n_err = int(((df["method"] == method) & df.get("error", pd.Series(dtype=str)).notna()).sum())

        if len(mdf) == 0:
            print(f"{method:<35} {'0':>5} {'--':>10} {'--':>9} {'--':>10} {'--':>9} {'--':>10} {'--':>10} {'--':>10} {'--':>9}")
            if n_err:
                sample = df.loc[(df["method"] == method) & df["error"].notna(), "error"].iloc[0]
                print(f"    ({n_err} errors, e.g.: {str(sample)[:100]})")
            continue

        fdps = mdf["fdp"].values
        tpps = mdf["tpp"].values
        discs = mdf["n_disc"].values
        rts = mdf["runtime_s"].values if "runtime_s" in mdf.columns else np.zeros(len(mdf))
        rss = mdf["peak_rss_mb"].values if "peak_rss_mb" in mdf.columns else np.zeros(len(mdf))
        ctrl = "Y" if fdps.mean() <= alpha + 0.005 else "N"

        label = method
        if method == "ko_knockpy" and "ko_type" in mdf.columns:
            types = mdf["ko_type"].dropna().unique()
            if len(types) == 1:
                label = f"Knockoffs ({types[0]}, lasso)"
        if method in ("bh", "by") and "test_mode" in mdf.columns:
            modes = mdf["test_mode"].dropna().unique()
            if len(modes) == 1:
                label = f"{'BH' if method == 'bh' else 'BY'} + {modes[0]}"

        print(
            f"{label:<35} {len(mdf):>5} "
            f"{fdps.mean():>10.4f} {fdps.std():>9.4f} "
            f"{tpps.mean():>10.4f} {tpps.std():>9.4f} "
            f"{discs.mean():>10.1f} {rts.mean():>10.2f} "
            f"{rss.mean():>10.1f} {ctrl:>9}"
        )

        detail_rows.append(dict(
            method=label, n_runs=len(mdf),
            mean_fdp=fdps.mean(), std_fdp=fdps.std(),
            mean_tpp=tpps.mean(), std_tpp=tpps.std(),
            mean_disc=discs.mean(), mean_runtime=rts.mean(),
            mean_rss_mb=rss.mean(),
            fdr_controlled=(fdps.mean() <= alpha + 0.005),
        ))

    print("=" * 120)
    return detail_rows


# =============================================================
# Main
# =============================================================

def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data_preprocessed"
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    pheno_model = sys.argv[3] if len(sys.argv) > 3 else "multiplicative_rr"
    run_subsample_arg = sys.argv[4] if len(sys.argv) > 4 else None
    snp_subsample_arg = sys.argv[5] if len(sys.argv) > 5 else None

    snp_subsample = int(snp_subsample_arg) if snp_subsample_arg and snp_subsample_arg.isdigit() else None

    cfg = dict(
        pheno_model=pheno_model,
        s=10,
        h2=0.3,
        prevalence=0.5,
        het_rr_range=(1.05, 1.25),
        alpha=0.1,
        seed0=42,
    )

    trex_cfg = dict(
        alpha=cfg["alpha"], pheno_model=pheno_model,
        K=20, L_factor=5, T_stop=-1, calib="CalibrateT", solver="LARS",
        posthoc_mode=False, stride_width=5, max_stale_strides=2,
        n_threads=1, seed=42,
    )
    bh_by_cfg = dict(alpha=cfg["alpha"], pheno_model=pheno_model)
    ko_cfg = dict(alpha=cfg["alpha"], seed=42)

    method_timeout = {
        "trex": 3600, "bh": 3600, "by": 3600,
        "ko_id": 600, "ko_knockpy": 7200,
    }
    methods = ["trex", "bh", "by", "ko_id", "ko_knockpy"]
    method_cfgs = {
        "trex": trex_cfg, "bh": bh_by_cfg, "by": bh_by_cfg,
        "ko_id": ko_cfg, "ko_knockpy": ko_cfg,
    }

    run_subsample_seed = int(os.environ.get("HAPNEST_SUBSEED", "123"))
    snp_subsample_seed = int(os.environ.get("HAPNEST_SNPSEED", "456"))

    candidate_ids = sorted([
        i for i in range(1, n_runs + 1)
        if (Path(data_dir) / f"run_{i}" / "data.npz").exists()
        or (Path(data_dir) / f"run_{i}" / "X_std.dat").exists()
        or (Path(data_dir) / f"run_{i}" / "X_raw.dat").exists()
    ])
    run_ids, run_info = choose_run_subset(candidate_ids, run_subsample_arg, run_subsample_seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path("results") / f"benchmark_{pheno_model}_s{cfg['s']}_a{cfg['alpha']}_{ts}"
    pheno_dir = outdir / "phenotypes"
    store = ResultStore(outdir)

    print("=" * 80)
    print("HAPNEST Benchmark (subprocess-isolated)")
    print("=" * 80)
    print(f"Data:          {data_dir}")
    print(f"Runs:          {len(run_ids)} (of {len(candidate_ids)} available)")
    print(f"SNP subsample: {snp_subsample}")
    print(f"Phenotype:     {pheno_model} s={cfg['s']} h2={cfg['h2']} "
          f"prev={cfg['prevalence']} het_rr={cfg['het_rr_range']}")
    print(f"Alpha:         {cfg['alpha']}")
    print(f"Methods:       {methods}")
    print(f"Timeouts:      {method_timeout}")
    print(f"T-Rex:         n_threads=1 solver=LARS calib=CalibrateT")
    print(f"Output:        {outdir}")
    print(f"Worker:        {WORKER_SCRIPT}")
    print()

    # Phase 0
    print("--- Phase 0: Generating phenotypes ---")
    generate_all_phenotypes(
        data_dir, run_ids, cfg, pheno_dir,
        snp_subsample=snp_subsample, snp_seed=snp_subsample_seed,
    )

    # Phase 1
    print("--- Phase 1: Running methods ---")
    total_jobs = len(run_ids) * len(methods)
    n_done = 0

    for rid in run_ids:
        if store.killed:
            break
        for method in methods:
            if store.killed:
                break

            n_done += 1
            result = run_worker(
                data_dir, pheno_dir, rid, method,
                snp_subsample, method_cfgs[method],
                timeout=method_timeout.get(method),
            )
            store.add(result)

            status = "OK" if "error" not in result else f"ERR: {result['error'][:50]}"
            fdp_s = f"FDP={result['fdp']:.3f}" if "fdp" in result else ""
            tpp_s = f"TPP={result['tpp']:.3f}" if "tpp" in result else ""
            rt_s = f"t={result['runtime_s']:.1f}s" if "runtime_s" in result else ""
            rss_s = f"RSS={result['peak_rss_mb']:.0f}MB" if "peak_rss_mb" in result else ""
            tqdm.write(
                f"  [{n_done:>4}/{total_jobs}] run {rid:>3} {method:<12} "
                f"{fdp_s:>10} {tpp_s:>10} {rt_s:>10} {rss_s:>12} {status}"
            )

        store.flush()

    # Phase 2
    df = pd.DataFrame(store.rows)
    if len(df) == 0:
        print("No results.")
        return

    detail_rows = print_summary(df, methods, cfg["alpha"])
    pd.DataFrame(detail_rows).to_csv(outdir / "summary.csv", index=False)
    store.save_meta(dict(
        config=cfg,
        trex_cfg={k: v for k, v in trex_cfg.items() if k not in ("alpha", "pheno_model")},
        methods=methods, method_timeouts=method_timeout,
        n_runs=len(run_ids), n_results=len(df),
        n_errors=int(df["error"].notna().sum()) if "error" in df.columns else 0,
        run_subsample=run_info, snp_subsample=snp_subsample,
        killed=store.killed,
    ))

    print(f"\nResults saved to {outdir}/")


if __name__ == "__main__":
    main()
