#!/usr/bin/env bash

#!/usr/bin/env bash
PROJECT_DIR="/Users/abhinav/abhi/dct"
SRC_TRAIN_FILE="${PROJECT_DIR}/data/yelp/sentiment.train.0"
TRG_TRAIN_FILE="${PROJECT_DIR}/data/yelp/sentiment.train.1"
SRC_DEV_FILE="${PROJECT_DIR}/data/yelp/sentiment.dev.0"
TRG_DEV_FILE="${PROJECT_DIR}/data/yelp/sentiment.dev.1"
SRC_TEST_FILE="${PROJECT_DIR}/data/yelp/sentiment.test.0"
TRG_TEST_FILE="${PROJECT_DIR}/data/yelp/sentiment.test.1"
SRC_DOM_SALIENCY_FILE="${PROJECT_DIR}/data/yelp/src_dom.attributes.txt"
TRG_DOM_SALIENCY_FILE="${PROJECT_DIR}/data/yelp/trg_dom.attributes.txt"
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