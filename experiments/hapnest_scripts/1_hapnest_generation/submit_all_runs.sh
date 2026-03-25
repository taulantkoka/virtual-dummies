#!/bin/bash
N_RUNS=30
SCRATCH_DIR=/work/scratch/tk33sawa/hapnest_mc

mkdir -p ${SCRATCH_DIR}/{outputs,configs,logs}
cp ~/hapnest/config_100k_template.yaml ${SCRATCH_DIR}/

echo "HAPNEST MC: ${N_RUNS} runs x 22 chromosomes"

for i in $(seq 1 $N_RUNS); do
    JOB_ID=$(sbatch \
        --export=RUN_ID=$i \
        --output=${SCRATCH_DIR}/logs/geno_run${i}_chr%a_%j.out \
        --error=${SCRATCH_DIR}/logs/geno_run${i}_chr%a_%j.err \
        ~/hapnest/slurm_geno_chr.sh | awk '{print $4}')
    echo "Run $i -> job $JOB_ID"
done

echo "Monitor: squeue -u $(whoami)"
