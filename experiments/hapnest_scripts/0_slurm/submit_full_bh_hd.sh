#!/bin/bash
#SBATCH -J full_bh_hd
#SBATCH --array=1-30
#SBATCH -n 1
#SBATCH --mem-per-cpu=4G
#SBATCH -t 1-00:00:00
#SBATCH -A project_id
#SBATCH -o /work/scratch/USER_ID/hapnest_mc_full/logs/full_ko_hd_%a_%j.out
#SBATCH -e /work/scratch/USER_ID/hapnest_mc_full/logs/full_ko_hd_%a_%j.err

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate trex

export LD_PRELOAD=$CONDA_PREFIX/lib/libjemalloc.so
export LD_LIBRARY_PATH=/shared/apps/gcc/11.2.0/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

RUN_ID=$SLURM_ARRAY_TASK_ID
DATA_DIR=/work/scratch/USER_ID/hapnest_mc_full/preprocessed
PHENO_DIR=/work/scratch/USER_ID/hapnest_mc_full/benchmark_results/multiplicative_rr_full/phenotypes
RESULT_DIR=/work/scratch/USER_ID/hapnest_mc_full/benchmark_results/multiplicative_rr_full/results

echo "=== BH-HD run ${RUN_ID} ==="
echo "Node: $(hostname) | Start: $(date)"

cd ~/VirtualDummyForwardSelection

CFG="{\"alpha\":0.1,\"pheno_model\":\"multiplicative_rr\",\"_data_dir\":\"${DATA_DIR}\",\"_run_id\":${RUN_ID}}"

python benchmark_worker.py \
    "${DATA_DIR}" \
    "${PHENO_DIR}" \
    "${RUN_ID}" \
    "bh_hd" \
    "" \
    "${CFG}" \
    > "${RESULT_DIR}/run_${RUN_ID}_bh_hd.json"

echo "Exit: $? | Done: $(date)"
