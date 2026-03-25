#!/bin/bash
#SBATCH -J preproc
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem-per-cpu = 24G
#SBATCH -t 24:00:00
#SBATCH -A project_id
#SBATCH --array = 1-100
#SBATCH -o /work/scratch/USER_ID/hapnest_mc_small/logs/preproc_%a_%j.out
#SBATCH -e /work/scratch/USER_ID/hapnest_mc_small/logs/preproc_%a_%j.err

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate trex

export OMP_NUM_THREADS=1

RUN_ID=$SLURM_ARRAY_TASK_ID

echo "=== Preprocessing run ${RUN_ID} ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

python ~/VirtualDummyForwardSelection/hapnest_scripts/2_preprocessing/preprocess_hapnest_small.py \
    /work/scratch/USER_ID/hapnest_mc_small/outputs \
    /work/scratch/USER_ID/hapnest_mc_small/preprocessed \
    30 \
    --run-id $RUN_ID

EXITCODE=$?
echo "Finished with exit code $EXITCODE at $(date)"
exit $EXITCODE
