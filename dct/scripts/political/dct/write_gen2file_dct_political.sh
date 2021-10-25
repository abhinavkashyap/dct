#!/usr/bin/env bash
PROJECT_DIR="/home/rkashyap/abhi/dct"
DUMP_DIR="/data3/abhinav/dct"
DATA_DIR=${DUMP_DIR}/"data"
PYTHON_FILE=${PROJECT_DIR}/"dct/utils/write_gen2file.py"
EXP_DIR=${DUMP_DIR}/"experiments"
EXP_NAMES=("[YELP_POLITICAL]" "[YELP_POLITICAL_2]" "[YELP_POLITICAL_3]" "[YELP_POLITICAL_4]" "[YELP_POLITICAL_5]")
index=1
FROM_FILE=${DATA_DIR}/"political/democratic.test"


for EXP_NAME in ${EXP_NAMES[@]};
do
    python ${PYTHON_FILE} \
    --checkpoints-dir ${EXP_DIR}/${EXP_NAME}/"checkpoints" \
    --hparams-file ${EXP_DIR}/${EXP_NAME}/"hparams.json" \
    --from-file ${FROM_FILE} \
    --out-file ${DUMP_DIR}/results/political/dct/greedy.1.txt.seed${index} \
    --greedy

    python ${PYTHON_FILE} \
    --checkpoints-dir ${EXP_DIR}/${EXP_NAME}/"checkpoints" \
    --hparams-file ${EXP_DIR}/${EXP_NAME}/"hparams.json" \
    --from-file ${FROM_FILE} \
    --out-file ${DUMP_DIR}/results/political/dct/ns.p60.1.txt.seed${index} \
    --nucleus-sampling \
    --top-p 0.6 \
    --temperature 0.4

    python ${PYTHON_FILE} \
    --checkpoints-dir ${EXP_DIR}/${EXP_NAME}/"checkpoints" \
    --hparams-file ${EXP_DIR}/${EXP_NAME}/"hparams.json" \
    --from-file ${FROM_FILE} \
    --out-file ${DUMP_DIR}/results/political/dct/ns.p90.1.txt.seed${index} \
    --nucleus-sampling \
    --top-p 0.9 \
    --temperature 0.4

    ((index++))
done