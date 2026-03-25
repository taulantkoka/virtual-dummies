#!/usr/bin/env python
"""Run ALL methods for one run, loading data just once."""
import json, sys, os, time
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(__file__))
from benchmark_worker import *

data_dir = sys.argv[1]
pheno_dir = sys.argv[2]
run_id = int(sys.argv[3])
methods_file = sys.argv[4]
res_dir = sys.argv[5]
cfg_json = sys.argv[6] if len(sys.argv) > 6 else '{}'

base_cfg = json.loads(cfg_json)
base_cfg.setdefault("alpha", 0.1)
base_cfg.setdefault("pheno_model", "multiplicative_rr")

methods = [m.strip() for m in open(methods_file) if m.strip()]

# Load data ONCE
pheno = load_phenotype(pheno_dir, run_id)
if pheno is None:
    for m in methods:
        json.dump({"run": run_id, "method": m, "error": "pheno not found"},
                  open(f"{res_dir}/run_{run_id}_{m}.json", "w"))
    sys.exit(0)

y, causal_idx, Xc_bundled, X_raw_bundled = pheno
mmap_fd = -1
_mm_holders = []

if Xc_bundled is not None:
    Xc, X_raw = Xc_bundled, X_raw_bundled
    load_mode = "bundled"
else:
    loaded = load_data(data_dir, run_id, snp_subsample=None)
    if loaded is None:
        sys.exit(1)
    Xc, X_raw, mmap_fd, load_mode, _mm_holders = loaded

    # For non-pread methods, load into RAM for speed
    if load_mode in ("memmap", "pread"):
        log(f"Loading full matrix into RAM...")
        t0 = time.perf_counter()
        Xc_ram = np.array(Xc, dtype=np.float64, order='F')
        X_raw_ram = np.array(X_raw, dtype=np.float64, order='F')
        log(f"Loaded in {time.perf_counter()-t0:.1f}s, RSS={get_peak_rss_mb():.0f}MB")
    else:
        Xc_ram, X_raw_ram = Xc, X_raw

n, p = Xc_ram.shape
log(f"run={run_id} n={n} p={p} s={len(causal_idx)} load={load_mode}")

for method in methods:
    cfg = base_cfg.copy()
    if method == "trex":
        cfg.update({"tFDR":0.1,"K":20,"L_factor":5,"T_stop":50,
                     "solver":"LARS","calib":"CalibrateT","n_threads":1})

    out_path = f"{res_dir}/run_{run_id}_{method}.json"
    log(f"--- {method} ---")

    t0 = time.perf_counter()
    try:
        # Use pread for trex if available, RAM for others
        if method == "trex" and mmap_fd >= 0:
            selected, extra = METHODS[method](Xc, X_raw, y, cfg, mmap_fd)
        else:
            selected, extra = METHODS[method](Xc_ram, X_raw_ram, y, cfg, -1)
    except Exception as e:
        dt = time.perf_counter() - t0
        result = {"run": run_id, "method": method, "error": str(e),
                  "runtime_s": round(dt,4), "peak_rss_mb": round(get_peak_rss_mb(),1),
                  "n": n, "p": p, "load_mode": load_mode}
        json.dump(result, open(out_path, "w"))
        log(f"  ERROR: {e} ({dt:.1f}s)")
        continue

    dt = time.perf_counter() - t0
    fdp, tpp = fdr_tpp(selected, causal_idx)
    result = {"run": run_id, "method": method, "fdp": fdp, "tpp": tpp,
              "n_disc": len(selected), "runtime_s": round(dt,4),
              "peak_rss_mb": round(get_peak_rss_mb(),1),
              "n": n, "p": p, "n_causal": len(causal_idx), "load_mode": load_mode}
    result.update(extra)
    json.dump(result, open(out_path, "w"))
    log(f"  FDP={fdp:.3f} TPP={tpp:.3f} disc={len(selected)} time={dt:.1f}s")

log("All methods done.")
