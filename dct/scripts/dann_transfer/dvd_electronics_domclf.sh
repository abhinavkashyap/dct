#!/usr/bin/env bash
PROJECT_DIR="/home/ubuntu/abhi/dct"
DUMP_DIR="/abhinav/dct"
MODEL_DIR=${DUMP_DIR}/"ftmodels"
DOMCLF_MODEL_NAME="dann_transfer_domclf.model"
DATA_DIR=${DUMP_DIR}/"data"

python ${PROJECT_DIR}/dct/evaluation/transfer/fasttext_binaryclf.py \
--class1-train-file ${DATA_DIR}/"mcauley_reviews/dvd.transfer.train" \
--class2-train-file ${DATA_DIR}/"mcauley_reviews/electronics.transfer.train" \
--class1-test-file ${DATA_DIR}/"mcauley_reviews/dvd.transfer.test" \
--class2-test-file ${DATA_DIR}/"mcauley_reviews/electronics.transfer.test" \
--model-dir ${MODEL_DIR} \
--model-name ${DOMCLF_MODEL_NAME}

python ${PROJECT_DIR}/dct/evaluation/transfer/fasttext_binaryclf_infer.py \
--ft-model-path ${MODEL_DIR}/${DOMCLF_MODEL_NAME} \
--sentences-file ${DATA_DIR}/"mcauley_reviews/electronics.transfer.test" \
--correct-label "__label__2"