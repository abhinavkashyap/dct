#!/usr/bin/env bash
PROJECT_DIR="/home/ubuntu/abhi/dct"
DUMP_DIR="/abhinav/dct"
DATA_DIR=${DUMP_DIR}/"data"
PYTHON_FILE=${PROJECT_DIR}/"dct/utils/write_results_table.py"
ROBERTA_BASE_CHECKPOINTS_DIR=${DUMP_DIR}/"roberta_cola_model/cola_distilroberta-base_25e/checkpoints"
ROBERTA_BASE_HPARAMS_FILE=${DUMP_DIR}/"roberta_cola_model/cola_distilroberta-base_25e/hparams.json"
SIM_MODEL=${DUMP_DIR}/"similarity_models/sim.pt"
SIM_SENTENCEPIECE_MODEL=${DUMP_DIR}/"similarity_models/sim.sp.30k.model"
MODEL_NAME="DCT_CONTRA"
DATA_NAME="YELP"
RESULTS_FOLDER=${DUMP_DIR}/"results/yelp/contra"
FROM_FILE=${DATA_DIR}/"yelp/sentiment.test.0"
DOM_CLF_FT_MODEL=${DUMP_DIR}/"ftmodels/yelp_domclf.model"
SEEDS=(1 2 3 4)

for seed in "${SEEDS[@]}";
do
python ${PYTHON_FILE} \
--from-file ${FROM_FILE} \
--to-file ${DUMP_DIR}/"results/yelp/contra/greedy.1.txt.seed${seed}" \
--clf-ft-model-path ${DOM_CLF_FT_MODEL} \
--sim-model-file ${SIM_MODEL} \
--sim-sentencepiece-model-file ${SIM_SENTENCEPIECE_MODEL} \
--cola-roberta-checkpoints-dir ${ROBERTA_BASE_CHECKPOINTS_DIR} \
--cola-roberta-json-file ${ROBERTA_BASE_HPARAMS_FILE} \
--src-dom-saliency-file ${DATA_DIR}/"yelp/src_dom.attributes.txt" \
--trg-dom-saliency-file ${DATA_DIR}/"yelp/trg_dom.attributes.txt" \
--model-name ${MODEL_NAME} \
--gen-method "greedy" \
--data-name ${DATA_NAME} \
--main-results-csv-filename ${RESULTS_FOLDER}/"main_results.csv" \
--constrain-results-csv-filename ${RESULTS_FOLDER}/"constrain_results.csv"
done

for seed in ${SEEDS[@]};
do
python ${PYTHON_FILE} \
--from-file ${FROM_FILE} \
--to-file ${DUMP_DIR}/"results/yelp/contra/ns.p60.1.txt.seed${seed}" \
--clf-ft-model-path ${DOM_CLF_FT_MODEL} \
--sim-model-file ${SIM_MODEL} \
--sim-sentencepiece-model-file ${SIM_SENTENCEPIECE_MODEL} \
--cola-roberta-checkpoints-dir ${ROBERTA_BASE_CHECKPOINTS_DIR} \
--cola-roberta-json-file ${ROBERTA_BASE_HPARAMS_FILE} \
--src-dom-saliency-file ${DATA_DIR}/"yelp/src_dom.attributes.txt" \
--trg-dom-saliency-file ${DATA_DIR}/"yelp/trg_dom.attributes.txt" \
--model-name ${MODEL_NAME} \
--gen-method "nsp60" \
--data-name ${DATA_NAME} \
--main-results-csv-filename ${RESULTS_FOLDER}/"main_results.csv" \
--constrain-results-csv-filename ${RESULTS_FOLDER}/"constrain_results.csv"
done

for seed in ${SEEDS[@]};
do
python ${PYTHON_FILE} \
--from-file ${FROM_FILE} \
--to-file ${DUMP_DIR}/"results/yelp/contra/ns.p90.1.txt.seed${seed}" \
--clf-ft-model-path ${DOM_CLF_FT_MODEL} \
--sim-model-file ${SIM_MODEL} \
--sim-sentencepiece-model-file ${SIM_SENTENCEPIECE_MODEL} \
--cola-roberta-checkpoints-dir ${ROBERTA_BASE_CHECKPOINTS_DIR} \
--cola-roberta-json-file ${ROBERTA_BASE_HPARAMS_FILE} \
--src-dom-saliency-file ${DATA_DIR}/"yelp/src_dom.attributes.txt" \
--trg-dom-saliency-file ${DATA_DIR}/"yelp/trg_dom.attributes.txt" \
--model-name ${MODEL_NAME} \
--gen-method "nsp90" \
--data-name ${DATA_NAME} \
--main-results-csv-filename ${RESULTS_FOLDER}/"main_results.csv" \
--constrain-results-csv-filename ${RESULTS_FOLDER}/"constrain_results.csv"
done