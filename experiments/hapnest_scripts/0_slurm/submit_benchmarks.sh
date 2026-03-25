#!/bin/bash
# submit_benchmarks.sh — HAPNEST multi-scale benchmark
# set -e

SCRATCH=/work/scratch/USER_ID/hapnest_mc
DATA_SMALL=${SCRATCH}_small/preprocessed
DATA_LARGE=${SCRATCH}_full/preprocessed
SCRIPTS=~/VirtualDummyForwardSelection
LOGS=${SCRATCH}/logs
RESULTS=${SCRATCH}/benchmark_results
ACCOUNT= project_id

mkdir -p $LOGS $RESULTS

# Helper: submit sbatch and extract numeric job ID (filters LUA plugin noise)
submit_sbatch() {
    local raw
    raw=$(sbatch "$@" 2>&1)
    echo "$raw" | grep -Eo '^[0-9]+' | head -1
}

echo "============================================"
echo "HAPNEST Benchmark — Full Submission"
echo "============================================"
echo ""

submit_experiment() {
    local LABEL=$1
    local DATA_DIR=$2
    local N_RUNS=$3
    local PHENO=$4
    local SNP_SUB=$5
    local H2=$6
    local PREV=$7
    local METHODS_STR=$8
    local MEM_GB=$9
    local WALLTIME=${10}

    local EXP_DIR=${RESULTS}/${LABEL}
    local PHENO_DIR=${EXP_DIR}/phenotypes
    local RES_DIR=${EXP_DIR}/results
    mkdir -p $PHENO_DIR $RES_DIR

    IFS=',' read -ra METHODS <<< "$METHODS_STR"
    local N_METHODS=${#METHODS[@]}
    local N_TASKS=$((N_RUNS * N_METHODS))

    if [ -z "$SNP_SUB" ]; then
        local CFG="{\"s\":10,\"h2\":${H2},\"prevalence\":${PREV},\"het_rr_range\":[1.05,1.25],\"seed0\":42}"
    else
        local CFG="{\"s\":10,\"h2\":${H2},\"prevalence\":${PREV},\"het_rr_range\":[1.05,1.25],\"seed0\":42,\"snp_subsample\":${SNP_SUB},\"snp_seed\":456}"
    fi

    local TREX_CFG="{\"alpha\":0.1,\"pheno_model\":\"${PHENO}\",\"K\":20,\"L_factor\":5,\"T_stop\":50,\"calib\":\"CalibrateT\",\"solver\":\"LARS\",\"posthoc_mode\":false,\"stride_width\":5,\"max_stale_strides\":2,\"n_threads\":1,\"seed\":42}"
    local BH_CFG="{\"alpha\":0.1,\"pheno_model\":\"${PHENO}\"}"
    local KO_CFG="{\"alpha\":0.1,\"seed\":42}"

    cat > ${EXP_DIR}/experiment.json << METAEOF
{
    "label": "${LABEL}",
    "data_dir": "${DATA_DIR}",
    "pheno_model": "${PHENO}",
    "snp_subsample": $([ -z "$SNP_SUB" ] && echo "null" || echo "$SNP_SUB"),
    "h2": ${H2},
    "prevalence": ${PREV},
    "methods": ["$(echo $METHODS_STR | sed 's/,/","/g')"],
    "n_runs": ${N_RUNS},
    "n_tasks": ${N_TASKS},
    "alpha": 0.1,
    "s": 10,
    "het_rr_range": [1.05, 1.25]
}
METAEOF

    printf '%s\n' "${METHODS[@]}" > ${EXP_DIR}/methods.txt

    echo "--- ${LABEL} ---"
    echo "    pheno=${PHENO} snp_sub=${SNP_SUB:-full} h2=${H2} prev=${PREV}"
    echo "    methods=${METHODS_STR} (${N_METHODS}×${N_RUNS}=${N_TASKS} tasks)"

    # Phase 0 memory
    local PHENO_MEM="4G"
    [ -z "$SNP_SUB" ] && PHENO_MEM="16G"

    # Phase 0: generate phenotypes
    local P0_ID
    P0_ID=$(submit_sbatch \
        --parsable \
        --job-name=p0_${LABEL} \
        --array=1-${N_RUNS} \
        -n 1 -c 2 \
        --mem-per-cpu=${PHENO_MEM} \
        -t 02:00:00 \
        -A ${ACCOUNT} \
        -o ${LOGS}/p0_${LABEL}_%a.out \
        -e ${LOGS}/p0_${LABEL}_%a.err \
        --wrap="
eval \"\$(\$HOME/miniconda3/bin/conda shell.bash hook)\"
conda activate trex
export LD_PRELOAD=$CONDA_PREFIX/lib/libjemalloc.so
module load gcc/11.2.0 openblas/0.3.18
export OMP_NUM_THREADS=1
cd ${SCRIPTS}
python generate_phenotype.py ${DATA_DIR} ${PHENO_DIR} \${SLURM_ARRAY_TASK_ID} ${PHENO} '${CFG}'
")

    if [ -z "$P0_ID" ]; then
        echo "    ERROR: Phase 0 submission failed"
        return 1
    fi
    echo "    Phase 0: job ${P0_ID} (array 1-${N_RUNS})"

    # Phase 1: run methods
    local P1_ID
    P1_ID=$(submit_sbatch \
        --parsable \
        --job-name=p1_${LABEL} \
        --dependency=afterok:${P0_ID} \
        --array=1-${N_TASKS} \
        -n 1 -c 2 \
        --mem-per-cpu=${MEM_GB}G \
        -t ${WALLTIME} \
        -A ${ACCOUNT} \
        -o ${LOGS}/p1_${LABEL}_%a.out \
        -e ${LOGS}/p1_${LABEL}_%a.err \
        --wrap="
eval \"\$(\$HOME/miniconda3/bin/conda shell.bash hook)\"
conda activate trex
export LD_PRELOAD=$CONDA_PREFIX/lib/libjemalloc.so
module load gcc/11.2.0 openblas/0.3.18
export OMP_NUM_THREADS=1

N_METHODS=${N_METHODS}
TASK_ID=\${SLURM_ARRAY_TASK_ID}
RUN_IDX=\$(( (TASK_ID - 1) / N_METHODS ))
METH_IDX=\$(( (TASK_ID - 1) % N_METHODS ))
RUN_ID=\$(( RUN_IDX + 1 ))

METHOD=\$(sed -n \"\$(( METH_IDX + 1 ))p\" ${EXP_DIR}/methods.txt)

case \$METHOD in
    trex)       METHOD_CFG='${TREX_CFG}' ;;
    bh|by)      METHOD_CFG='${BH_CFG}' ;;
    ko_id|ko_knockpy) METHOD_CFG='${KO_CFG}' ;;
    *)          METHOD_CFG='{}' ;;
esac

SNP_ARG='${SNP_SUB}'

echo \"=== Task \${TASK_ID}: run=\${RUN_ID} method=\${METHOD} ===\"
echo \"Node: \$(hostname)  Start: \$(date)\"

cd ${SCRIPTS}
python benchmark_worker.py \\
    ${DATA_DIR} \\
    ${PHENO_DIR} \\
    \${RUN_ID} \\
    \${METHOD} \\
    \"\${SNP_ARG}\" \\
    \"\${METHOD_CFG}\" \\
    > ${RES_DIR}/run_\${RUN_ID}_\${METHOD}.json

echo \"Done: \$(date)\"
")

    if [ -z "$P1_ID" ]; then
        echo "    ERROR: Phase 1 submission failed"
        return 1
    fi
    echo "    Phase 1: job ${P1_ID} (array 1-${N_TASKS}, after ${P0_ID})"
    echo ""
}


# ============================================================
# Small: p=2k, 100 runs, prev=0.5
# ============================================================
for PHENO in multiplicative_rr linear liability_binary; do
    H2=0.3; [ "$PHENO" = "linear" ] && H2=0.15
    submit_experiment "${PHENO}_2k" "$DATA_SMALL" 100 "$PHENO" "2000" "$H2" "0.5" \
        "trex,bh,by,ko_id,ko_knockpy" "8" "1-00:00:00"
done

# ============================================================
# Medium: p=35k, 100 runs, prev=0.5
# ============================================================
for PHENO in multiplicative_rr linear liability_binary; do
    H2=0.3; [ "$PHENO" = "linear" ] && H2=0.15
    submit_experiment "${PHENO}_35k" "$DATA_SMALL" 100 "$PHENO" "35000" "$H2" "0.5" \
        "trex,bh,by,ko_knockpy" "16" "1-00:00:00"
done

# ============================================================
# Large: p=394k, 30 runs, prev=0.1
# ============================================================
for PHENO in multiplicative_rr linear liability_binary; do
    H2=0.3; [ "$PHENO" = "linear" ] && H2=0.15
    submit_experiment "${PHENO}_full" "$DATA_LARGE" 30 "$PHENO" "" "$H2" "0.1" \
        "trex,bh,by,ko_knockpy" "64" "1-00:00:00"
done

echo "============================================"
echo "All experiments submitted."
echo "  Small  (2k):  3 × 100 × 5 = 1500 tasks"
echo "  Medium (35k): 3 × 100 × 4 = 1200 tasks"
echo "  Large (394k): 3 ×  30 × 4 =  360 tasks"
echo "  Total: 3060 tasks"
echo ""
echo "After completion, merge:"
echo "  for d in ${RESULTS}/*/; do"
echo "    python ${SCRIPTS}/benchmark_merge.py \"\$d\""
echo "  done"
echo "============================================"