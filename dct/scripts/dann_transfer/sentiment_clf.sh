#!/usr/bin/env bash
# Trains sentiment classifier for dvd and electronics domain

PROJECT_DIR="/home/ubuntu/abhi/dct"
DUMP_DIR="/abhinav/dct/"
DATA_DIR=${DUMP_DIR}/"data/mcauley_reviews"
EXP_DIR=${DUMP_DIR}/"experiments"
PYTHON_FILE=${PROJECT_DIR}/"dct/utils/fasttext_clf.py"

python ${PYTHON_FILE} \
--train-file ${DATA_DIR}/"dvd_electronics.sentimentclf.train" \
--dev-file ${DATA_DIR}/"dvd_electronics.sentimentclf.dev" \
--test-file ${DATA_DIR}/"dvd_electronics.sentimentclf.test" \
--model-dir ${EXP_DIR}/"dvd_electronics_sentimentclf" \
--model-name dvd_electronics.model
