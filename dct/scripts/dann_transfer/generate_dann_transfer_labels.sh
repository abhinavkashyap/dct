#!/usr/bin/env bash

#!/usr/bin/env bash
PROJECT_DIR="/home/ubuntu/abhi/dct"
DUMP_DIR="/abhinav/dct"
SRC_TRAIN_FILE="${DUMP_DIR}/data/mcauley_reviews/dvd.transfer.train"
TRG_TRAIN_FILE="${DUMP_DIR}/data/mcauley_reviews/electronics.transfer.train"
SRC_DEV_FILE="${DUMP_DIR}/data/mcauley_reviews/dvd.transfer.dev"
TRG_DEV_FILE="${DUMP_DIR}/data/mcauley_reviews/electronics.transfer.dev"
SRC_TEST_FILE="${DUMP_DIR}/data/mcauley_reviews/dvd.transfer.test"
TRG_TEST_FILE="${DUMP_DIR}/data/mcauley_reviews/electronics.transfer.test"
SRC_DOM_SALIENCY_FILE="${DUMP_DIR}/data/mcauley_reviews/src_dom.attributes.txt"
TRG_DOM_SALIENCY_FILE="${DUMP_DIR}/data/mcauley_reviews/trg_dom.attributes.txt"
GENERATE_YELP_LABELS_PYTHON=${PROJECT_DIR}/"dct/utils/generate_constraints.py"

######################
# NEGATIVE SENT
######################
python ${GENERATE_YELP_LABELS_PYTHON} \
--filename ${SRC_TRAIN_FILE} \
--out-label-filename ${SRC_TRAIN_FILE}".labels" \
--avg-length 10 \
--src-dom-saliency-file ${SRC_DOM_SALIENCY_FILE} \
--trg-dom-saliency-file ${TRG_DOM_SALIENCY_FILE}


python ${GENERATE_YELP_LABELS_PYTHON} \
--filename ${SRC_DEV_FILE} \
--out-label-filename ${SRC_DEV_FILE}".labels" \
--avg-length 10 \
--src-dom-saliency-file ${SRC_DOM_SALIENCY_FILE} \
--trg-dom-saliency-file ${TRG_DOM_SALIENCY_FILE}

python ${GENERATE_YELP_LABELS_PYTHON} \
--filename ${SRC_TEST_FILE} \
--out-label-filename ${SRC_TEST_FILE}".labels" \
--avg-length 10 \
--src-dom-saliency-file ${SRC_DOM_SALIENCY_FILE} \
--trg-dom-saliency-file ${TRG_DOM_SALIENCY_FILE}

######################
# POSITIVE SENT
######################
python ${GENERATE_YELP_LABELS_PYTHON} \
--filename ${TRG_TRAIN_FILE} \
--out-label-filename ${TRG_TRAIN_FILE}".labels" \
--avg-length 10 \
--src-dom-saliency-file ${SRC_DOM_SALIENCY_FILE} \
--trg-dom-saliency-file ${TRG_DOM_SALIENCY_FILE}

python ${GENERATE_YELP_LABELS_PYTHON} \
--filename ${TRG_DEV_FILE} \
--out-label-filename ${TRG_DEV_FILE}".labels" \
--avg-length 10 \
--src-dom-saliency-file ${SRC_DOM_SALIENCY_FILE} \
--trg-dom-saliency-file ${TRG_DOM_SALIENCY_FILE}


python ${GENERATE_YELP_LABELS_PYTHON} \
--filename ${TRG_TEST_FILE} \
--out-label-filename ${TRG_TEST_FILE}".labels" \
--avg-length 10 \
--src-dom-saliency-file ${SRC_DOM_SALIENCY_FILE} \
--trg-dom-saliency-file ${TRG_DOM_SALIENCY_FILE}