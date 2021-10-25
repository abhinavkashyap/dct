#!/usr/bin/env bash
PROJECT_DIR="/home/rkashyap/abhi/dct"
DUMP_DIR="/data3/abhinav/dct"
DATA_DIR=${DUMP_DIR}/"data"
PYTHON_FILE=${PROJECT_DIR}/"dct/utils/write_gen2file.py"
EXP_DIR=${DUMP_DIR}/"experiments"
EXP_NAMES=("[IMDB_DCT]" "[IMDB_DCT_2]" "[IMDB_DCT_3]" "[IMDB_DCT_4]")
index=1
FROM_FILE=${DATA_DIR}/"imdb/test.neg"


for EXP_NAME in ${EXP_NAMES[@]};
do
    python ${PYTHON_FILE} \
    --checkpoints-dir ${EXP_DIR}/${EXP_NAME}/"checkpoints" \
    --hparams-file ${EXP_DIR}/${EXP_NAME}/"hparams.json" \
    --from-file ${FROM_FILE} \
    --out-file ${DUMP_DIR}/results/imdb/dct/greedy.1.txt.seed${index} \
    --greedy

    python ${PYTHON_FILE} \
    --checkpoints-dir ${EXP_DIR}/${EXP_NAME}/"checkpoints" \
    --hparams-file ${EXP_DIR}/${EXP_NAME}/"hparams.json" \
    --from-file ${FROM_FILE} \
    --out-file ${DUMP_DIR}/results/imdb/dct/ns.p60.1.txt.seed${index} \
    --nucleus-sampling \
    --top-p 0.6 \
    --temperature 0.4

    python ${PYTHON_FILE} \
    --checkpoints-dir ${EXP_DIR}/${EXP_NAME}/"checkpoints" \
    --hparams-file ${EXP_DIR}/${EXP_NAME}/"hparams.json" \
    --from-file ${FROM_FILE} \
    --out-file ${DUMP_DIR}/results/imdb/dct/ns.p90.1.txt.seed${index} \
    --nucleus-sampling \
    --top-p 0.9 \
    --temperature 0.4

    ((index++))
done