# HAPNEST GWAS Benchmark (Section 6.6, Table 1)

Reproduces the GWAS benchmark on realistic linkage-disequilibrium data simulated
with [HAPNEST](https://github.com/simlab-bioinf/hapnest) (Wharrie et al., 2023).

## Overview

The pipeline has four phases:

1. **Genotype generation** (`1_genotype_generation/`) — HAPNEST simulates
   genotypes from European-ancestry reference panels, one chromosome at a time,
   followed by PLINK QC (MAF > 1%, call rate > 95%, HWE p > 1e-6).

2. **Preprocessing** (`2_preprocessing/`) — LD pruning via single-linkage
   clustering at |corr| > 0.7, keeping one representative SNP per cluster.
   Outputs Fortran-order memmap files (`X_std.dat`, `X_raw.dat`).

3. **Phenotype generation** (`3_phenotype/`) — Binary case-control phenotypes
   under a multiplicative relative risk model.

4. **Benchmark** (`4_benchmark/`) — Each method runs on each replicate
   independently. Results collected as JSON, merged into summary tables.

## Directory Layout

```
hapnest/
├── 1_genotype_generation/
│   ├── slurm_geno_chr.sh              # Per-chromosome genotype + PLINK QC
│   ├── submit_all_runs.sh             # Launch all runs (30 × 22 chr)
│   └── config_100k_template.yaml      # HAPNEST config template
│
├── 2_preprocessing/
│   ├── preprocess_hapnest_small.py    # Small-scale (chr 1, pandas_plink)
│   ├── preprocess_hapnest_full.py     # Full-scale (all 22 chr, bed_reader)
│   ├── run_preprocess_small.sh        # Slurm wrapper for small-scale
│   └── run_preprocess_full.sh         # Slurm wrapper for full-scale
│
├── 3_phenotype/
│   └── generate_phenotype.py          # Case-control phenotype generation
│
├── 4_benchmark/
│   ├── benchmark_worker.py            # Core: one method, one replicate
│   ├── benchmark_allmethod_worker.py  # All methods, one replicate
│   ├── benchmark_merge.py             # Collect JSONs → summary CSV
│   ├── submit_benchmarks.sh           # Main orchestrator (all scales)
│   ├── submit_full_bh_stream.sh       # BH marginal (full-scale, streaming)
│   ├── submit_full_by_stream.sh       # BY marginal (full-scale, streaming)
│   ├── submit_full_bh_hd.sh           # BH high-dimensional (full-scale)
│   ├── submit_full_by_hd.sh           # BY high-dimensional (full-scale)
│   └── submit_full_ko_hd.sh           # Knockoffs-HD (full-scale)
│
└── README.md                          # this file
```

## Data Availability

Quality controlled HAPNEST genotypes (PLINK format) are available through Harvard Dataverse (https://dataverse.harvard.edu/dataverse/taulantkoka):

    https://doi.org/10.7910/DVN/RZ3FZT  (small-scale)
    https://doi.org/10.7910/DVN/KQHP5C  (full-scale 1-10)
    https://doi.org/10.7910/DVN/IJ4MVT  (full-scale 11-20)
    https://doi.org/10.7910/DVN/MDOFTV  (full-scale 21-30)

To preprocess after downloading:
```bash
python 2_preprocessing/preprocess_hapnest_small.py <raw_dir> <out_dir> 100
python 2_preprocessing/preprocess_hapnest_full.py <raw_dir> <out_dir> 30
```

## Quick Start (with downloaded raw data)

```bash
# 1. Build vd_selectors
cd /path/to/virtual-dummies
pip install -e .

# 2. Install Python dependencies for competing methods
pip install statsmodels scikit-learn knockpy pandas_plink bed_reader

# 3. Preprocess (small-scale example)
python 2_preprocessing/preprocess_hapnest_small.py \
    data/hapnest_raw_small \
    data/hapnest_preprocessed_small \
    100

# 4. Generate phenotypes (100 replicates)
for i in $(seq 1 100); do
    python 3_phenotype/generate_phenotype.py \
        data/hapnest_preprocessed_small \
        results/hapnest_small/phenotypes \
        $i \
        multiplicative_rr \
        '{"s":10,"prevalence":0.5,"het_rr_range":[1.05,1.25],"seed0":42}'
done

# 5. Run all methods for one replicate
echo -e "trex\nbh\nby\nko_knockpy" > methods.txt
python 4_benchmark/benchmark_allmethod_worker.py \
    data/hapnest_preprocessed_small \
    results/hapnest_small/phenotypes \
    1 \
    methods.txt \
    results/hapnest_small/results \
    '{"alpha":0.1,"pheno_model":"multiplicative_rr"}'

# 6. Merge results into summary table
python 4_benchmark/benchmark_merge.py results/hapnest_small
```

## Methods Implemented (in `benchmark_worker.py`)

| Key | Method | Description |
|-----|--------|-------------|
| `trex` | **VD-T-Rex** | Virtual-dummy T-Rex selector (this paper) |
| `bh` | BH-CA | Cochran-Armitage + Benjamini-Hochberg |
| `by` | BY-CA | Cochran-Armitage + Benjamini-Yekutieli |
| `bh_hd` | BH-HD | Sample-split screening + OLS + BH (Barber & Candes, 2019) |
| `by_hd` | BY-HD | Sample-split screening + OLS + BY |
| `ko_hd` | Knockoffs-HD | Sample-split screening + fixed-X knockoffs |
| `ko_knockpy` | Model-X Knockoffs | Full knockpy (Spector & Janson, 2022) |
| `bh_stream` | BH-CA (streaming) | Column-streaming marginal tests for full-scale |
| `by_stream` | BY-CA (streaming) | Column-streaming marginal tests for full-scale |

## Experimental Settings

### Small-scale (Table 1a)
- **Genotypes**: HAPNEST, chr 1, European ancestry
- **n** = 10,000 individuals, **p** ~ 31,000 SNPs after QC + LD pruning
- **Phenotype**: multiplicative RR, prevalence 0.5, s = 10 causal SNPs
- **Het. RR** ~ Unif(1.05, 1.25)
- **Replicates**: 100
- **VD-T-Rex**: K = 20, L = 5p, CalibrateT, spherical dummies

### Full-scale (Table 1b)
- **Genotypes**: HAPNEST, all 22 autosomes, European ancestry
- **n** = 100,000 individuals, **p** ~ 394,000 SNPs after QC + LD pruning
- **Phenotype**: multiplicative RR, prevalence 0.1, s = 10 causal SNPs
- **Het. RR** ~ Unif(1.05, 1.25)
- **Replicates**: 30
- **VD-T-Rex**: K = 20, L = 5p, CalibrateT, spherical dummies, pread I/O
- **Memory-mapped access**: `X_std.dat` accessed via `MMapMatrix` (pread)

## Preprocessing Details

LD pruning uses single-linkage clustering on pairwise absolute correlations:
- Correlations estimated on a random subsample of 3,000 rows (re-standardized
  so that `Xc_sub.T @ Xc_sub` gives valid correlations in [-1, 1])
- Threshold: |corr| > 0.7 -> merge into same cluster
- One representative per cluster (random selection, seeded)
- Final matrix: column-centered, unit-L2 normalized, Fortran-order float64

This yields approximately:
- Small-scale: ~31,000 SNPs from chromosome 1
- Full-scale: ~394,000 SNPs across all 22 autosomes

Both preprocessing scripts produce identical output format:
```
run_<id>/
├── X_std.dat       # standardized design matrix (n × p, F-order, float64)
├── X_raw.dat       # raw 0/1/2 dosages (n × p, F-order, float64)
├── meta.json       # n_samples, p_pruned_total, per_chromosome stats
└── kept_snps.txt   # SNP IDs of retained representatives
```

## Slurm Scripts (HPC)

All Slurm scripts use `USER_ID` and `project_id` placeholders.
**Replace these with your cluster username and account before submitting.**

### Preprocessing
```bash
# Small-scale (100 runs, chr 1):
sbatch --array=1-100 2_preprocessing/run_preprocess_small.sh

# Full-scale (30 runs, all 22 chr):
sbatch --array=1-30 2_preprocessing/run_preprocess_full.sh
```

### Benchmarks

The main orchestrator (`submit_benchmarks.sh`) handles phenotype generation
and method dispatch across all scales. Individual full-scale method scripts
are provided for running specific methods separately:

| Script | Method | Wall time | Memory |
|--------|--------|-----------|--------|
| `submit_full_bh_stream.sh` | BH-CA (streaming) | 1 day | 4 GB |
| `submit_full_by_stream.sh` | BY-CA (streaming) | 1 day | 4 GB |
| `submit_full_bh_hd.sh` | BH-HD | 1 day | 4 GB |
| `submit_full_by_hd.sh` | BY-HD | 1 day | 4 GB |
| `submit_full_ko_hd.sh` | Knockoffs-HD | 1 day | 4 GB |

VD-T-Rex at full scale is dispatched via `submit_benchmarks.sh` with 64 GB
memory allocation to accommodate pread I/O.

## Full Regeneration (from scratch)

Requires:
- [HAPNEST](https://github.com/simlab-bioinf/hapnest) (Julia)
- 1000 Genomes + HGDP reference panels (see HAPNEST docs)
- PLINK 1.9 and PLINK 2.0
- Python: `bed_reader` (full-scale) or `pandas_plink` (small-scale)

```bash
# 1. Generate genotypes (30 runs × 22 chromosomes)
#    Edit paths in 1_genotype_generation/config_100k_template.yaml
#    and 1_genotype_generation/slurm_geno_chr.sh
cd 1_genotype_generation/
bash submit_all_runs.sh

# 2. Preprocess
sbatch --array=1-100 2_preprocessing/run_preprocess_small.sh   # small
sbatch --array=1-30  2_preprocessing/run_preprocess_full.sh    # full

# 3. Generate phenotypes + run benchmarks
cd 4_benchmark/
bash submit_benchmarks.sh

# 4. After all jobs complete, merge results
for d in /path/to/results/*/; do
    python benchmark_merge.py "$d"
done
```

## Output Format

Each benchmark run produces a JSON file:
```json
{
    "run": 1,
    "method": "trex",
    "fdp": 0.0,
    "tpp": 0.6,
    "n_disc": 6,
    "runtime_s": 149.2,
    "peak_rss_mb": 16384.0,
    "n": 100000,
    "p": 394212,
    "n_causal": 10,
    "load_mode": "pread",
    "T_stop": 3,
    "L": 1971060,
    "L_calibrated": 1971060
}
```

`benchmark_merge.py` aggregates these into `summary.csv` (Table 1).

## References

* **HAPNEST**: Wharrie, S., et al. (2023). HAPNEST: efficient, large-scale generation and evaluation of synthetic datasets of whole genome sequence data. *Bioinformatics*, 39(1).
* **T-Rex Selector**: Machkour, J., Muma, M., & Palomar, D. P. (2025). The terminating-random experiments selector: Fast high-dimensional variable selection with false discovery rate control. *Signal Processing*, 231, 109894.
* **BH Correction**: Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B (Methodological)*.
* **BY Correction**: Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *The Annals of Statistics*, 29(4).
* **High-Dimensional Selective Inference (BH-HD/BY-HD)**: Barber, R. F., & Candès, E. J. (2019). A knockoff filter for high-dimensional selective inference. *The Annals of Statistics*, 47(5).
* **Knockoff Filter**: Barber, R. F., & Candès, E. J. (2015). Controlling the false discovery rate via knockoffs. *The Annals of Statistics*, 43(5).
* **Model-X Knockoffs**: Candès, E., Fan, Y., Janson, L., & Lv, J. (2018). Panning for Gold: 'Model-X' Knockoffs for High Dimensional Controlled Variable Selection. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 80(3).
* **Powerful Knockoffs (knockpy)**: Spector, A., & Janson, L. (2022). Powerful knockoffs via minimizing reconstructability. *The Annals of Statistics*, 50(1).