# Virtual Dummies for FDR-Controlled Variable Selection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

C++ library with Python bindings for scalable FDR-controlled variable selection
via virtual dummy features. Implements the VD-LARS, VD-OMP, VD-AFS, and
VD-AFS-Logistic forward selectors, plus the T-Rex selector for automatic
calibration.

**Paper**: T. Koka, J. Machkour, M. Muma. *"Virtual Dummies: Enabling Scalable
FDR-Controlled Variable Selection via Sequential Sampling of Null Features."*
Submitted to JMLR, 2026.

## Installation

```bash
git clone https://github.com/taulokoka/virtual-dummies.git
cd virtual-dummies
pip install .
```

This builds the `vd_selectors` C++ extension via scikit-build-core + pybind11.
Eigen headers are vendored in `extern/` (or fetched automatically). A
system BLAS (OpenBLAS, MKL, or macOS Accelerate) is used if available.

### macOS with Xcode 17+

If you get a `__hash_memory` symbol error at import time:

```bash
export CC=/usr/bin/clang CXX=/usr/bin/clang++
pip install .
```

### Development install

```bash
pip install -e .
```

## Quick Start

### Variable selection with VD-LARS

```python
import numpy as np
from vd_selectors import VD_LARS, VDOptions, VDDummyLaw

# Generate data
rng = np.random.default_rng(42)
n, p, s = 200, 1000, 5
X = rng.standard_normal((n, p))
X -= X.mean(axis=0); X /= np.linalg.norm(X, axis=0, keepdims=True)
X = np.asfortranarray(X)

beta_true = np.zeros(p)
beta_true[:s] = 1.0
y = X @ beta_true + 0.5 * rng.standard_normal(n)
y -= y.mean()

# Configure VD-LARS
opt = VDOptions()
opt.T_stop = 10          # stop after 10 virtual dummies are realized
opt.seed = 123
opt.dummy_law = VDDummyLaw.Spherical

# Run with L = 5p virtual dummies
num_dummies = 5 * p
solver = VD_LARS(X, y, num_dummies, opt)
solver.run(T=5)  # advance until 5 dummies realized

# Inspect results
beta_hat = np.array(solver.beta_real())
selected = np.where(np.abs(beta_hat) > 1e-12)[0]
print("Selected features:", selected)
print("Active set:", [af for af in solver.active_features()])
```

### FDR-controlled selection with T-Rex

```python
from vd_selectors import (
    TRexSelector, TRexOptions, SolverType, CalibMode, VDDummyLaw
)

opt = TRexOptions()
opt.tFDR = 0.1              # target FDR level
opt.K = 20                  # number of random experiments
opt.L_factor = 10           # L = 10p virtual dummies per experiment
opt.solver = SolverType.LARS
opt.calib = CalibMode.CalibrateT   # calibrate T, fix L
opt.dummy_law = VDDummyLaw.Spherical
opt.seed = 42
opt.verbose = False

trex = TRexSelector(opt)
result = trex.run(X, y)

selected = np.array(result.selected_var)
print(f"Selected {len(selected)} variables at FDR <= {opt.tFDR}")
print(f"T* = {result.T_stop}, v* = {result.v_thresh:.3f}")
```

### Memory-mapped data (large-scale GWAS)

For datasets that don't fit in RAM, use `MMapMatrix` with `pread` I/O:

```python
from vd_selectors import MMapMatrix, VD_LARS, VDOptions

# Load a Fortran-order float64 memmap file
mm = MMapMatrix("X_std.dat", nrows=100000, ncols=394000)
print(mm)  # <MMapMatrix 100000x394000 (295562 MB) ro>

# Use the file descriptor for pread-based column streaming
opt = VDOptions()
opt.mmap_fd = mm.fileno()
opt.mmap_block_cols = 2000   # read 2000 columns per pread call
opt.T_stop = 50
opt.seed = 42

X_view = mm.as_array()  # zero-copy numpy view
y = np.load("y.npy")

solver = VD_LARS(X_view, y, num_dummies=5 * 394000, opt)
solver.run(T=3)
```

## API Reference

### Solvers

All solvers share the same constructor signature and accessor methods.

| Class | Algorithm | Description |
|-------|-----------|-------------|
| `VD_LARS` | LARS | Equiangular direction, gamma-step (default, recommended) |
| `VD_OMP` | OMP | Greedy argmax, full OLS refit each step |
| `VD_AFS` | AFS | Adaptive forward stepwise with blending parameter rho |
| `VD_AFS_Logistic` | AFS-GLM | AFS with logistic link (binary response) |

**Constructor** (all solvers):
```python
solver = VD_LARS(X, y, num_dummies, options)
```
- `X`: `(n, p)` Fortran-order float64 array (column-centered, unit-L2 normalized)
- `y`: `(n,)` float64 array (centered)
- `num_dummies`: number of virtual dummies (typically `L * p`)
- `options`: `VDOptions` instance

**Methods** (all solvers):

| Method | Returns | Description |
|--------|---------|-------------|
| `run(T)` | `MatrixXd` | Advance until T dummies realized (warm-start) |
| `beta_real()` | `(p,)` array | Current coefficient vector for real features |
| `beta_view_copy()` | `(p,)` array | Copy of full beta (includes normalization) |
| `active_features()` | list of tuples | `[("real", j), ("dummy", k), ...]` |
| `active_indices()` | int array | Indices of active real features |
| `vd_corr()` | `(L,)` array | Virtual dummy correlations with residual |
| `vd_proj()` | `(m, L)` array | Virtual dummy projection coefficients |
| `vd_stick()` | `(L,)` array | Remaining stick lengths (spherical law) |
| `is_dummy_realized()` | int array | Binary mask of realized dummies |
| `normx()` | `(p,)` array | Column norms (if standardize=True) |
| `basis_size()` | int | Current orthonormal basis dimension |
| `n_samples()` | int | Number of rows |
| `n_features()` | int | Number of real features |
| `num_dummies()` | int | Total virtual dummies (L) |
| `num_realized_dummies()` | int | Number realized so far |

### VDOptions

```python
opt = VDOptions()
opt.T_stop = 100          # max dummies to realize (ceiling)
opt.max_vd_proj = 100     # max basis vectors for VD projection
opt.eps = 1e-12           # numerical zero
opt.standardize = False   # True: center + unit-L2 normalize X internally
opt.seed = 0              # RNG seed for dummy generation
opt.dummy_law = VDDummyLaw.Spherical  # or VDDummyLaw.Gaussian
opt.rho = 1.0             # AFS blending parameter (ignored by LARS/OMP)
opt.mmap_fd = -1          # file descriptor for pread I/O (-1 = disabled)
opt.mmap_block_cols = 0   # columns per pread block (0 = auto)
```

### TRexSelector

Automatic FDR-controlled variable selection via the T-Rex procedure.

```python
opt = TRexOptions()
opt.tFDR = 0.1            # target false discovery rate
opt.K = 20                # number of random experiments
opt.L_factor = 10         # L = L_factor * p virtual dummies
opt.T_stop = -1           # -1 = auto (min(L, n/2))
opt.solver = SolverType.LARS
opt.calib = CalibMode.CalibrateT
opt.dummy_law = VDDummyLaw.Spherical
opt.seed = 42
opt.verbose = True
opt.n_threads = 0         # 0 = auto (uses OpenMP if available)

# Calibration control
opt.stride_width = 1      # T increments between FDP checks
opt.posthoc_mode = False  # True: run all K to T_stop, then grid-search
opt.max_stale_strides = 3 # early-stop if no new reals for this many strides

trex = TRexSelector(opt)
result = trex.run(X, y)
```

**TRexResult** fields:

| Field | Type | Description |
|-------|------|-------------|
| `selected_var` | int array | Indices of selected variables |
| `v_thresh` | float | Optimal voting threshold |
| `T_stop` | int | Calibrated T |
| `num_dummies` | int | L used |
| `L_calibrated` | int | L after calibration (if CalibrateBoth) |
| `V` | float array | Voting threshold grid |
| `FDP_hat_mat` | `(T, |V|)` matrix | Estimated FDP surface |
| `Phi_mat` | `(T, p)` matrix | Voting matrix |
| `Phi_prime` | `(p,)` array | Corrected voting proportions |
| `K` | int | Number of experiments used |

### Calibration Modes

| Mode | Calibrates | Description |
|------|-----------|-------------|
| `FixedTL` | v only | User specifies T and L, grid-search over v |
| `CalibrateT` | T, v | Fix L = L_factor * p, search (T, v) with early stopping |
| `CalibrateL` | L | Scan L = p, 2p, ... until FDP condition met, then fix T=1 |
| `CalibrateBoth` | L, T, v | Calibrate L first, then search (T, v) |

## Preprocessing Requirements

The solvers expect:
1. **X**: column-centered, unit-L2 normalized, Fortran-order (`np.asfortranarray`)
2. **y**: centered (zero mean)

```python
def preprocess(X, y, eps=1e-12):
    X = np.asarray(X, dtype=np.float64)
    X -= X.mean(axis=0, keepdims=True)
    X /= np.linalg.norm(X, axis=0, keepdims=True).clip(min=eps)
    X = np.asfortranarray(X)
    y = np.asarray(y, dtype=np.float64)
    y -= y.mean()
    return X, y
```

## Repo Structure

```
virtual-dummies/
├── CMakeLists.txt          # Build: vd_selectors (always) + fig7_benchmark (opt-in)
├── pyproject.toml          # scikit-build-core config
├── src/                    # C++ source
│   ├── vd_common.hpp       # Shared types, options, GEMV helpers
│   ├── vd_base.hpp/cpp     # Base class: VD pool, basis, Cholesky, accessors
│   ├── vd_lars.hpp/cpp     # VD-LARS solver
│   ├── vd_omp.hpp/cpp      # VD-OMP solver
│   ├── vd_afs.hpp/cpp      # VD-AFS solver
│   ├── vd_afs_logistic.hpp/cpp  # VD-AFS with logistic link
│   ├── trex.hpp/cpp        # TRexSelector (calibration + solver dispatch)
│   ├── memory_mapped_eigen_matrix.hpp  # MMapMatrix for pread I/O
│   └── bindings.cpp        # pybind11 bindings
├── vd_selectors/
│   └── __init__.py         # Python package entry point
├── extern/
│   └── eigen-5.0.0/        # Vendored Eigen headers
├── experiments/            # Paper figure reproduction (see experiments/README.md)
│   ├── fig3_distributional_equivalence.py
│   ├── fig4_fdr_control_vd_vs_ad.py
│   ├── fig5_universality_diagnostics.py
│   ├── fig6_spherical_vs_gaussian.py
│   ├── fig7_memory_runtime_benchmark.cpp
│   ├── AD_LARS.py          # Python AD baseline
│   ├── helpers.pyx/.py     # Cython/Python helpers for AD_LARS
│   ├── tlars_core/         # Armadillo AD-LARS (for fig7 only)
│   └── hapnest/            # GWAS benchmark pipeline (Table 1)
└── tests/
```

## Reproducing Paper Experiments

See [`experiments/README.md`](experiments/README.md) for complete instructions on reproducing Figures 3–7 and Table 1.


## License

[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

## References

- **T-Rex Selector**: Machkour, J., Muma, M., & Palomar, D. P. (2025). The terminating-random experiments selector: Fast high-dimensional variable selection with false discovery rate control. *Signal Processing*, 231, 109894.
- **LARS**: Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least angle regression. *The Annals of Statistics*, 32(2).
- **OMP**: Pati, Y. C., Rezaiifar, R., & Krishnaprasad, P. S. (1993). Orthogonal matching pursuit: Recursive function approximation with applications to wavelet decomposition. *Proceedings of the 27th Asilomar Conference on Signals, Systems and Computers*.
- **Adaptive Forward Stepwise (AFS)**: Zhang, I., & Tibshirani, R. (2026). Adaptive Forward Stepwise: A Method for High Sparsity Regression. *Journal of Machine Learning Research*, 27(35).
  year={2026},

## Citation

If you use this software or the Virtual Dummy construction in your research, please cite:

```bibtex
@article{koka2026virtualdummies,
  title={Virtual Dummies: Enabling Scalable {FDR}-Controlled Variable Selection via Sequential Sampling of Null Features},
  author={Koka, Taulant and Machkour, Jasin and Muma, Michael},
  journal={arXiv preprint arXiv:26XX.XXXXX},
  year={2026},
}