#!/usr/bin/env bash
PROJECT_DIR="/home/ubuntu/abhi/dct"
DUMP_DIR="/abhinav/dct/"
DATA_DIR=${DUMP_DIR}/"data"
PYTHON_FILE=${PROJECT_DIR}/"dct/utils/write_gen2file.py"
EXP_DIR=${DUMP_DIR}/"experiments"
EXP_NAMES=("[IMDB_CLF_CONTRA]" "[IMDB_CLF_CONTRA_2]" "[IMDB_CLF_CONTRA_4]" "[IMDB_CLF_CONTRA_4]" "[IMDB_CLF_CONTRA_5]")
RESULTS_DIR=${DUMP_DIR}/results/imdb/clf_contra
FROM_FILE=${DATA_DIR}/"imdb/test.neg"
index=1

for EXP_NAME in ${EXP_NAMES[@]};
do
    python ${PYTHON_FILE} \
    --checkpoints-dir ${EXP_DIR}/${EXP_NAME}/"checkpoints" \
    --hparams-file ${EXP_DIR}/${EXP_NAME}/"hparams.json" \
    --from-file ${FROM_FILE} \
    --out-file ${RESULTS_DIR}/greedy.1.txt.seed${index} \
    --greedy \
    --is-contrastive

    python ${PYTHON_FILE} \
    --checkpoints-dir ${EXP_DIR}/${EXP_NAME}/"checkpoints" \
    --hparams-file ${EXP_DIR}/${EXP_NAME}/"hparams.json" \
    --from-file ${FROM_FILE} \
    --out-file ${RESULTS_DIR}/ns.p60.1.txt.seed${index} \
    --nucleus-sampling \
    --top-p 0.6 \
    --temperature 0.4 \
    --is-contrastive


    python ${PYTHON_FILE} \
    --checkpoints-dir ${EXP_DIR}/${EXP_NAME}/"checkpoints" \
    --hparams-file ${EXP_DIR}/${EXP_NAME}/"hparams.json" \
    --from-file ${FROM_FILE} \
    --out-file ${RESULTS_DIR}/ns.p90.1.txt.seed${index} \
    --nucleus-sampling \
    --top-p 0.9 \
    --temperature 0.4 \
    --is-contrastive

    ((index++))

done
