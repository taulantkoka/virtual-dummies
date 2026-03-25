"""
HAPNEST Preprocessing — Memory-efficient two-pass approach
Pass 1: cluster (3k rows only), record rep indices
Pass 2: read rep columns one chr at a time, write directly to memmap

FIX: validates that all chromosomes share the same sample IDs.
     If they don't, restricts to the intersection of common IIDs.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from bed_reader import open_bed

import warnings
warnings.filterwarnings("ignore")

EPS = 1e-12
MAX_N_FOR_CLUSTERING = 3000
CHROMOSOMES = list(range(1, 23))


def center_unitL2(X, eps=EPS):
    X = X - X.mean(axis=0, keepdims=True)
    n2 = np.linalg.norm(X, axis=0, keepdims=True).clip(min=eps)
    return X / n2


class UnionFind:
    def __init__(self, n):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int32)
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry: return
        if self.rank[rx] < self.rank[ry]: rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]: self.rank[rx] += 1


def cluster_components_threshold(Xc, *, threshold=0.7, block_size=2000):
    n, p = Xc.shape
    uf = UnionFind(p)
    total_edges = 0
    for start in range(0, p, block_size):
        end = min(start + block_size, p)
        C_block = Xc[:, start:end].T @ Xc
        for j in range(end - start):
            C_block[j, start + j] = 0.0
        rows, cols = np.where(np.abs(C_block) > threshold)
        for r, c in zip(rows, cols):
            global_r = start + r
            if global_r < c:
                uf.union(global_r, c)
                total_edges += 1
    roots = np.array([uf.find(j) for j in range(p)], dtype=np.int32)
    clusters = {}
    for j in range(p):
        clusters.setdefault(int(roots[j]), []).append(j)
    n_clusters = len(clusters)
    singletons = sum(1 for c in clusters.values() if len(c) == 1)
    max_cluster = max(len(c) for c in clusters.values())
    print(f"        {total_edges} pairs |corr|>{threshold} -> "
          f"{n_clusters} clusters (singletons: {singletons}, max: {max_cluster})",
          flush=True)
    return clusters


def choose_representatives(clusters, *, mode, rng):
    reps = []
    for members in clusters.values():
        if mode == "first":
            reps.append(int(min(members)))
        elif mode == "random":
            reps.append(int(members[rng.integers(len(members))]))
    return np.array(sorted(reps), dtype=int)


# ----------------------------------------------------------------
# Pass 0 (new): collect IID lists for every chromosome in this run
# ----------------------------------------------------------------
def collect_chr_iids(run_dir, run_id, chroms=CHROMOSOMES):
    """Return {chrom: (n, iid_list)} for every chromosome that exists."""
    info = {}
    for chrom in chroms:
        bed_path = run_dir / f"run_{run_id}_chr-{chrom}.bed"
        if not bed_path.exists():
            continue
        with open_bed(str(bed_path)) as bed:
            info[chrom] = (bed.iid_count, bed.iid.tolist())
    return info


def resolve_common_samples(chr_iid_info):
    """
    Given {chrom: (n, iid_list)}, figure out the common set of IIDs.

    Returns
    -------
    common_iids : list[str]          ordered intersection
    row_indices : dict[int, np.array] per-chrom row indices into bed file
    mismatch    : bool                True if any chromosome differed
    """
    if not chr_iid_info:
        return [], {}, False

    all_sets = [set(iids) for _, iids in chr_iid_info.values()]
    common = all_sets[0]
    for s in all_sets[1:]:
        common &= s

    # Check if all chromosomes already agree
    counts = [n for n, _ in chr_iid_info.values()]
    mismatch = len(set(counts)) > 1 or any(n != len(common) for n in counts)

    # Use ordering from the first chromosome
    first_chrom = min(chr_iid_info.keys())
    _, first_iids = chr_iid_info[first_chrom]
    common_ordered = [iid for iid in first_iids if iid in common]

    # Build per-chromosome row index arrays
    row_indices = {}
    for chrom, (n, iids) in chr_iid_info.items():
        if n == len(common_ordered) and iids == common_ordered:
            row_indices[chrom] = None  # means "use all rows, no subsetting needed"
        else:
            iid_to_pos = {iid: i for i, iid in enumerate(iids)}
            row_indices[chrom] = np.array(
                [iid_to_pos[iid] for iid in common_ordered], dtype=np.intp
            )

    return common_ordered, row_indices, mismatch


def cluster_chromosome(
    run_dir, run_id, chrom, *,
    corr_threshold, rng, block_size=2000, rep_mode="random",
    row_idx=None,
):
    """
    Pass 1: cluster using ≤3k rows, return metadata + rep column indices.

    Parameters
    ----------
    row_idx : np.ndarray | None
        If not None, only these rows are read from the bed file so that
        all chromosomes use the same set of individuals.
    """
    prefix = str(run_dir / f"run_{run_id}_chr-{chrom}")
    bed_path = prefix + ".bed"
    if not Path(bed_path).exists():
        return None

    with open_bed(bed_path) as bed:
        n_bed = bed.iid_count
        p_chr = bed.sid_count
        snp_ids = bed.sid.tolist()

        # Read subset of rows for clustering
        if row_idx is not None:
            sub_rows = row_idx
            n = len(row_idx)
        else:
            sub_rows = None
            n = n_bed

        n_clust = min(n, MAX_N_FOR_CLUSTERING)
        if n_clust < n:
            clust_rows = rng.choice(n, size=n_clust, replace=False)
            if sub_rows is not None:
                clust_rows = sub_rows[clust_rows]
        else:
            clust_rows = sub_rows  # None means all rows

        X_sub = bed.read(index=(clust_rows, None)).astype(np.float64)
        np.nan_to_num(X_sub, copy=False, nan=0.0)

    print(f"    chr {chrom}: {n_bed} x {p_chr}, clustering on {X_sub.shape[0]} rows",
          flush=True)

    Xc_sub = center_unitL2(X_sub)
    clusters = cluster_components_threshold(
        Xc_sub, threshold=corr_threshold, block_size=block_size
    )
    reps = choose_representatives(clusters, mode=rep_mode, rng=rng)

    return {
        "chrom": chrom,
        "n": n_bed,
        "p_raw": p_chr,
        "p_pruned": len(reps),
        "n_clusters": len(clusters),
        "reps": reps,
        "snp_ids": snp_ids,
        "rep_snp_ids": [snp_ids[j] for j in reps],
    }


def write_chr_to_memmap(run_dir, run_id, chrom, reps, X_raw_mm, X_std_mm,
                        col_offset, row_idx=None):
    """
    Pass 2: read only the representative columns from this chromosome's
    bed file and write them into the pre-allocated memmap files.
    """
    bed_path = str(run_dir / f"run_{run_id}_chr-{chrom}.bed")
    with open_bed(bed_path) as bed:
        block_size = 2000
        for c0 in range(0, len(reps), block_size):
            c1 = min(c0 + block_size, len(reps))
            cols = reps[c0:c1].tolist()
            block = bed.read(index=(row_idx, cols)).astype(np.float64)
            np.nan_to_num(block, copy=False, nan=0.0)
            X_raw_mm[:, col_offset + c0:col_offset + c1] = block

            # Standardize
            block_std = block - block.mean(axis=0, keepdims=True)
            norms = np.linalg.norm(block_std, axis=0, keepdims=True)
            norms[norms < EPS] = 1.0
            block_std /= norms
            X_std_mm[:, col_offset + c0:col_offset + c1] = block_std


def process_one_run(
    run_id, base_dir, out_dir, *,
    corr_threshold=0.7, seed0=42, block_size=2000, rep_mode="random",
):
    run_dir = Path(base_dir) / f"run_{run_id}"
    if not run_dir.exists():
        print(f"  Run {run_id}: directory not found, skipping")
        return False

    seed = int(seed0 + run_id * 10_007)
    rng = np.random.default_rng(seed)

    # --- Pass 0: Collect IIDs, validate sample counts ---
    print(f"\n  Run {run_id}: collecting sample IDs...", flush=True)
    chr_iid_info = collect_chr_iids(run_dir, run_id)
    if not chr_iid_info:
        print(f"  Run {run_id}: no chromosome files found, skipping")
        return False

    common_iids, row_indices, mismatch = resolve_common_samples(chr_iid_info)
    n = len(common_iids)

    if mismatch:
        counts = {c: info[0] for c, info in chr_iid_info.items()}
        print(f"  Run {run_id}: WARNING — sample counts differ across chromosomes: {counts}")
        print(f"  Using intersection of {n} common IIDs")
    else:
        print(f"  Run {run_id}: all chromosomes have {n} samples")

    # --- Pass 1: Cluster each chromosome ---
    print(f"  Run {run_id}: clustering...", flush=True)
    chr_info = []
    all_rep_snp_ids = []
    p_total = 0

    for chrom in CHROMOSOMES:
        if chrom not in chr_iid_info:
            continue
        ri = row_indices.get(chrom)
        ci = cluster_chromosome(
            run_dir, run_id, chrom,
            corr_threshold=corr_threshold, rng=rng,
            block_size=block_size, rep_mode=rep_mode,
            row_idx=ri,
        )
        if ci is None:
            continue
        chr_info.append(ci)
        all_rep_snp_ids.extend(ci["rep_snp_ids"])
        p_total += ci["p_pruned"]

    if p_total == 0:
        print(f"  Run {run_id}: no SNPs survived pruning, skipping")
        return False

    print(f"  Run {run_id}: {p_total} SNPs after pruning across "
          f"{len(chr_info)} chromosomes", flush=True)

    # --- Pass 2: Write representative columns to memmap ---
    run_out = out_dir / f"run_{run_id}"
    run_out.mkdir(parents=True, exist_ok=True)

    X_raw_mm = np.memmap(
        run_out / "X_raw.dat", dtype="float64", mode="w+",
        shape=(n, p_total), order="F",
    )
    X_std_mm = np.memmap(
        run_out / "X_std.dat", dtype="float64", mode="w+",
        shape=(n, p_total), order="F",
    )

    col_offset = 0
    for ci in chr_info:
        ri = row_indices.get(ci["chrom"])
        write_chr_to_memmap(
            run_dir, run_id, ci["chrom"], ci["reps"],
            X_raw_mm, X_std_mm, col_offset, row_idx=ri,
        )
        col_offset += ci["p_pruned"]

    X_raw_mm.flush()
    X_std_mm.flush()
    del X_raw_mm, X_std_mm

    # --- Save metadata ---
    meta = {
        "run_id": run_id,
        "seed": seed,
        "n_samples": n,
        "p_raw_total": sum(ci["p_raw"] for ci in chr_info),
        "p_pruned_total": p_total,
        "n_chromosomes": len(chr_info),
        "corr_threshold": corr_threshold,
        "rep_mode": rep_mode,
        "max_n_for_clustering": MAX_N_FOR_CLUSTERING,
        "iid_mismatch": mismatch,
        "per_chromosome": [
            {
                "chrom": ci["chrom"],
                "p_raw": ci["p_raw"],
                "p_pruned": ci["p_pruned"],
                "n_clusters": ci["n_clusters"],
            }
            for ci in chr_info
        ],
    }
    with open(run_out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(run_out / "kept_snps.txt", "w") as f:
        f.write("\n".join(all_rep_snp_ids) + "\n")

    print(f"  Run {run_id}: done — {n} x {p_total}, saved to {run_out}", flush=True)
    return True


# =============================================================
# Main
# =============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAPNEST full-scale preprocessing")
    parser.add_argument("base_dir", help="Directory with run_*/run_*_chr-*.bed files")
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument("n_runs", type=int, help="Total number of runs")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Process a single run (for SLURM array jobs)")
    parser.add_argument("--corr-threshold", type=float, default=0.7)
    parser.add_argument("--seed0", type=int, default=42)
    parser.add_argument("--rep-mode", default="random", choices=["random", "first"])
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HAPNEST Preprocessing — Full-scale (all chromosomes, memmap)")
    print("=" * 60)
    print(f"Input:           {base_dir}")
    print(f"Output:          {out_dir}")
    print(f"corr_threshold:  {args.corr_threshold}")
    print(f"rep_mode:        {args.rep_mode}")
    print(f"seed0:           {args.seed0}")
    print(f"max_n_for_clust: {MAX_N_FOR_CLUSTERING}")
    print("=" * 60)

    if args.run_id is not None:
        # Single run mode (SLURM array)
        ok = process_one_run(
            args.run_id, base_dir, out_dir,
            corr_threshold=args.corr_threshold,
            seed0=args.seed0,
            rep_mode=args.rep_mode,
        )
        sys.exit(0 if ok else 1)
    else:
        # Process all runs
        success = failed = 0
        for rid in range(1, args.n_runs + 1):
            ok = process_one_run(
                rid, base_dir, out_dir,
                corr_threshold=args.corr_threshold,
                seed0=args.seed0,
                rep_mode=args.rep_mode,
            )
            if ok:
                success += 1
            else:
                failed += 1
        print(f"\nDone! Success: {success}, Failed: {failed}")
        print(f"Output: {out_dir}/run_*/{{X_std.dat, X_raw.dat, meta.json, kept_snps.txt}}")