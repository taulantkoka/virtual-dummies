#!/usr/bin/env python
"""
Merge benchmark results from per-task JSON files.

Usage:
    python benchmark_merge.py <results_dir> [alpha]

Reads:  <results_dir>/results/run_*_*.json
Writes: <results_dir>/all_results.csv
        <results_dir>/per_run.csv
        <results_dir>/summary.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    results_dir = Path(sys.argv[1])
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1

    # Find all result JSONs
    res_dir = results_dir / "results"
    if not res_dir.exists():
        res_dir = results_dir  # fallback: results directly in dir

    json_files = sorted(res_dir.glob("run_*_*.json"))
    if not json_files:
        print(f"No result files found in {res_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} result files")

    # Load all
    all_rows = []
    for jf in json_files:
        try:
            with open(jf) as f:
                r = json.load(f)
            all_rows.append(r)
        except Exception as e:
            print(f"  Warning: failed to load {jf}: {e}")

    df = pd.DataFrame(all_rows)
    df.to_csv(results_dir / "all_results.csv", index=False)
    print(f"Wrote all_results.csv ({len(df)} rows)")

    # Pivoted per-run
    pivot_rows = []
    for rid, grp in df.groupby("run"):
        row = {"run": rid}
        if "n" in grp.columns:
            row["n"] = grp["n"].iloc[0]
        if "p" in grp.columns:
            row["p"] = grp["p"].iloc[0]
        for _, r in grp.iterrows():
            m = r["method"]
            for col in ["fdp", "tpp", "n_disc", "runtime_s", "peak_rss_mb"]:
                if col in r and pd.notna(r[col]):
                    row[f"{m}_{col}"] = r[col]
            if "error" in r and pd.notna(r.get("error")):
                row[f"{m}_error"] = r["error"]
            for col in ["T_stop", "L", "L_calibrated", "test_mode", "ko_type", "load_mode"]:
                if col in r and pd.notna(r.get(col)):
                    row[f"{m}_{col}"] = r[col]
        pivot_rows.append(row)
    pd.DataFrame(pivot_rows).to_csv(results_dir / "per_run.csv", index=False)
    print(f"Wrote per_run.csv ({len(pivot_rows)} runs)")

    # Summary
    methods = df["method"].unique()
    print(f"\n{'=' * 120}")
    print(f"SUMMARY ({len(df)} results across {df['run'].nunique()} runs)")
    print(f"{'=' * 120}")
    print(
        f"{'Method':<35} {'Runs':>5} {'Mean FDP':>10} {'Std FDP':>9} "
        f"{'Mean TPP':>10} {'Std TPP':>9} {'Mean disc':>10} "
        f"{'Mean t(s)':>10} {'Mean RSS':>10} {'FDR ctrl':>9}"
    )
    print("-" * 120)

    detail_rows = []
    for method in sorted(methods):
        mdf = df[(df["method"] == method) & df["fdp"].notna()] if "fdp" in df.columns else pd.DataFrame()
        n_err = int(((df["method"] == method) & df.get("error", pd.Series(dtype=str)).notna()).sum()) if "error" in df.columns else 0

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
    pd.DataFrame(detail_rows).to_csv(results_dir / "summary.csv", index=False)
    print(f"Wrote summary.csv")


if __name__ == "__main__":
    main()
