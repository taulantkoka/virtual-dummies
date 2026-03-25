# Reproducing Paper Experiments

Scripts for reproducing Figures 3–7 and Table 1 from the paper.

## Setup

```bash
# 1. Install vd_selectors (from repo root)
pip install -e .

# 2. Install experiment dependencies
pip install numpy scipy matplotlib joblib tqdm pandas

# 3. Build Cython helpers for AD-LARS
cd experiments/
pip install cython
python setup.py build_ext --inplace
# If skipped, the pure-Python helpers.py fallback is used automatically.

# 4. (macOS Xcode 17+ only) If you get __hash_memory symbol errors:
export CC=/usr/bin/clang CXX=/usr/bin/clang++
pip install -e .
```

## Figure–Script Mapping

| Section | Output  | Script                              |
|---------|---------|-------------------------------------|
| §6.1    | Fig 3   | `fig3_distributional_equivalence.py` |
| §6.2    | Fig 4   | `fig4_fdr_control_vd_vs_ad.py`      |
| §6.3    | Fig 5   | `fig5_universality_diagnostics.py`   |
| §6.4    | Fig 6   | `fig6_spherical_vs_gaussian.py`     |
| §6.5    | Fig 7   | `fig7_memory_runtime_benchmark.cpp` |
| §6.6    | Table 1 | `hapnest/` pipeline                |

## Running

All Python scripts are self-contained — just run them from the `experiments/` directory:

```bash
cd experiments/

# Fig 3: Distributional equivalence of VD vs AD correlations
python fig3_distributional_equivalence.py

# Fig 4: FDR control and power comparison
python fig4_fdr_control_vd_vs_ad.py

# Fig 5: Universality diagnostics (conditional CLT convergence)
python fig5_universality_diagnostics.py

# Fig 6: Spherical vs Gaussian dummy law heatmaps
python fig6_spherical_vs_gaussian.py
```

Results are saved to `experiments/results/<fig_name>/`.

### Fig 7: Memory/runtime benchmark (C++)

This is the only C++ experiment. It requires Armadillo (for the AD-LARS baseline).

```bash
# Build from repo root:
cmake -B build -DBUILD_BENCHMARKS=ON
cmake --build build --target fig7_benchmark

# Run (skip AD runs exceeding 100 GB):
TLARS_MEM_CAP_GB=100 VD_MEM_CAP_GB=100 ./build/fig7_benchmark
```

Output: `results/bench_p_signal_n10000_<timestamp>/raw.csv`

### Table 1: HAPNEST GWAS benchmark

See [`hapnest/README.md`](hapnest/README.md) for the full pipeline.
Preprocessed data is available at https://doi.org/10.5281/zenodo.XXXXXXX.

## File Layout

```
experiments/
├── fig3_distributional_equivalence.py
├── fig4_fdr_control_vd_vs_ad.py
├── fig5_universality_diagnostics.py
├── fig6_spherical_vs_gaussian.py
├── fig7_memory_runtime_benchmark.cpp
│
├── AD_LARS.py          # Augmented-dummy LARS (Python baseline for Fig 3–5)
├── helpers.pyx         # Cython: Cholesky rank-1 update + gamma computation
├── helpers.py          # Pure-Python fallback (same API)
├── setup.py            # Cython build: python setup.py build_ext --inplace
│
├── tlars_core/         # AD-LARS C++ baseline (Fig 7 only)
│   └── src/
│       ├── TLARS_Solver.hpp / .cpp
│       └── arma_cereal.hpp
│
└── hapnest/            # GWAS benchmark pipeline (Table 1)
    ├── 1_genotype_generation/
    ├── 2_preprocessing/
    ├── 3_phenotype/
    ├── 4_benchmark/
    └── README.md
```

## Script Details

### Fig 3 — Distributional Equivalence (§6.1)

Compares order-statistic trajectories and ECDFs of |dummy–residual correlations|
between AD-LARS (explicit `[X|D]` augmentation) and VD-LARS (virtual dummies).
Fixed design, 2000 MC replicates. Uses both `AD_LARS` and `VD_LARS` from
`vd_selectors`.

### Fig 4 — FDR Control and Power (§6.2)

Sweeps SNR × L, comparing AD-T-Rex and VD-T-Rex. Both methods use the **same Python T-Rex calibration loop** (`trex_select`); the only difference is the solver backend. K solvers are created once and warm-started across T.

### Fig 5 — Universality Diagnostics (§6.3)

Tests the conditional CLT (Lemma 6) by measuring KS distance and Wasserstein-1 distance of dummy projections onto LARS basis vectors vs N(0,1), across different dummy distributions (Gaussian, Rademacher, Pareto, t, exponential, lognormal) and sample sizes. Uses `AD_LARS` with `track_basis=True` and `stop="steps"`.

### Fig 6 — Spherical vs Gaussian Dummy Law (§6.4)

Compares FDP and TPP surfaces over a grid of (alpha, T) for spherical vs Gaussian base laws. Uses `VD_LARS` from `vd_selectors` with `VDDummyLaw.Spherical` and `VDDummyLaw.Gaussian`.

### Fig 7 — Memory and Runtime Benchmark (§6.5)

C++ benchmark comparing AD-LARS (Armadillo, explicit augmentation) vs VD-LARS (Eigen, virtual dummies) in terms of peak RSS and wall-clock time across p = 1k–100k. Each (method, p, L, rep) runs in a **forked child process** for clean memory measurement. Linux/macOS only (`fork()` + `getrusage()`).

### AD_LARS.py — Unified AD Baseline

Two stopping modes and optional basis tracking are controlled via parameters:

```python
from AD_LARS import AD_LARS

# Stop when T dummies entered (Fig 3, Fig 4):
solver = AD_LARS(X_augmented, y, num_dummies=L)
solver.run(T=5)

# Stop after K LARS steps, track basis (Fig 5):
solver = AD_LARS(X_augmented, y, num_dummies=L, track_basis=True)
solver.run(T=100, stop="steps")
```

Both modes are warm-start capable.

## Platform Notes

**macOS (Xcode 17+)**: Use `export CC=/usr/bin/clang CXX=/usr/bin/clang++`
before building to avoid `__hash_memory` symbol errors.

**Windows**: Fig 7 (`fork()` + `getrusage()`) is Linux/macOS only. All Python
experiments (Fig 3–6) should work but are untested on Windows.

**Linux HPC**: The HAPNEST Slurm scripts contain cluster-specific paths.
See `hapnest/README.md`.

## References

* **T-Rex Selector**: Machkour, J., Muma, M., & Palomar, D. P. (2025). The terminating-random experiments selector: Fast high-dimensional variable selection with false discovery rate control. *Signal Processing*, 231, 109894.
* **LARS**: Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least angle regression. *The Annals of Statistics*, 32(2).