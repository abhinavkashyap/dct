#!/usr/bin/env bash
PROJECT_DIR="/home/rkashyap/abhi/dct"
DUMP_DIR="/data3/abhinav/dct/"
DATA_DIR=${DUMP_DIR}/"data"
OUTPUT_DIR=${DATA_DIR}/"processed_blitzer_reviews"
PYTHON_FILE=${PROJECT_DIR}/"dct/utils/blitzer_reviews_clean.py"
PROCESSED_DVD_FOLDER=${OUTPUT_DIR}/"dvd"
PROCESSED_ELECTRONICS_FOLDER=${OUTPUT_DIR}/"electronics"

# Craete directories if it does not exist yet
mkdir -p ${PROCESSED_DVD_FOLDER}
mkdir -p ${PROCESSED_ELECTRONICS_FOLDER}

python ${PYTHON_FILE} \
--dataset_folder_path ${DATA_DIR}/"blitzer_reviews/dvd" \
--output_folder ${PROCESSED_DVD_FOLDER} \
--file_prefix "dvd"

python ${PYTHON_FILE} \
--dataset_folder_path ${DATA_DIR}/"blitzer_reviews/electronics" \
--output_folder ${PROCESSED_ELECTRONICS_FOLDER} \
--file_prefix "electronics"