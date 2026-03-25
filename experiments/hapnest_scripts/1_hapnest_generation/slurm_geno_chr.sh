#!/bin/bash
#SBATCH -J hapnest_geno
#SBATCH --array=1-22
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem-per-cpu=2G
#SBATCH -t 1-
#SBATCH -A p0020087

HAPNEST_DIR=$HOME/hapnest
SCRATCH_DIR=/work/scratch/tk33sawa/hapnest_mc
DATA_DIR=$HOME/hapnest/data

CHR=$SLURM_ARRAY_TASK_ID
THREADS=$SLURM_CPUS_PER_TASK

module purge
module load gcc/11.2.0 gsl/2.7 openblas/0.3.18

export LD_LIBRARY_PATH="$HOME/tools/libplinkio/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export DATA_DIR
export SCRIPT_DIR=$HAPNEST_DIR

OUTDIR=${SCRATCH_DIR}/outputs/run_${RUN_ID}
CONFIG_DIR=${SCRATCH_DIR}/configs
CONFIG=${CONFIG_DIR}/config_run${RUN_ID}_chr${CHR}.yaml

mkdir -p $OUTDIR $CONFIG_DIR

cp ${SCRATCH_DIR}/config_100k_template.yaml $CONFIG
sed -i "s|__SEED__|${RUN_ID}|g"          $CONFIG
sed -i "s|__CHR__|${CHR}|g"              $CONFIG
sed -i "s|__RUNID__|${RUN_ID}|g"         $CONFIG
sed -i "s|__OUTDIR__|${OUTDIR}|g"        $CONFIG
sed -i "s|__DATADIR__|${DATA_DIR}|g"     $CONFIG

echo "=== Run ${RUN_ID} | Chr ${CHR} | Threads ${THREADS} ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

cd $HAPNEST_DIR
julia --project=. --threads $THREADS run_program.jl --genotype --config $CONFIG
EXITCODE=$?

if [ $EXITCODE -ne 0 ]; then
    echo "ERROR: genotype generation failed for run ${RUN_ID} chr ${CHR}"
    exit $EXITCODE
fi

RAW=${OUTDIR}/run_${RUN_ID}_chr-${CHR}

if [ -f ${RAW}.bed ]; then
    echo "Running PLINK QC..."
    $HOME/tools/plink1/plink \
        --bfile $RAW \
        --maf 0.01 \
        --geno 0.05 \
        --hwe 1e-6 \
        --make-bed \
        --out ${RAW}_qc \
        --silent

    if [ -f ${RAW}_qc.bed ]; then
        NSNPS_RAW=$(wc -l < ${RAW}.bim)
        NSNPS_QC=$(wc -l < ${RAW}_qc.bim)
        echo "QC: $NSNPS_RAW -> $NSNPS_QC SNPs"
        mv ${RAW}_qc.bed ${RAW}.bed
        mv ${RAW}_qc.bim ${RAW}.bim
        mv ${RAW}_qc.fam ${RAW}.fam
        rm -f ${RAW}_qc.*
    fi
fi

rm -f ${RAW}_0.bed ${RAW}_0.bim ${RAW}_0.fam ${RAW}_0.nosex ${RAW}_0.log
rm -f ${OUTDIR}/*.nosex ${OUTDIR}/*.log
rm -f $CONFIG

echo "Done: run ${RUN_ID} chr ${CHR} at $(date)"
exit $EXITCODE
