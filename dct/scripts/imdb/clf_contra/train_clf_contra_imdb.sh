#!/usr/bin/env bash

PROJECT_DIR="/home/ubuntu/abhi/dct"
DUMP_DIR="/abhinav/dct/"
EXP_DIR=${DUMP_DIR}/"experiments"
DATA_DIR=${DUMP_DIR}/"data"
LR_GEN=0.001
LR_DISC=0.0001
LR_AE=0.001
BATCH_SIZE=64
GPUS=(0 1 2 3)
GAN_GP_LAMBDA=0.1
HIDDEN_SIZE=300
TRAIN_PROPORTION=1
N_EPOCHS=35
FT_MODEL=${DUMP_DIR}/"ftmodels/imdb_domclf.model"
ROBERTA_BASE_CHECKPOINTS_DIR=${DUMP_DIR}/"roberta_cola_model/cola_distilroberta-base_25e/checkpoints"
ROBERTA_BASE_HPARAMS_FILE=${DUMP_DIR}/"roberta_cola_model/cola_distilroberta-base_25e/hparams.json"
SIM_MODEL=${DUMP_DIR}/"similarity_models/sim.pt"
SIM_SENTENCEPIECE_MODEL=${DUMP_DIR}/"similarity_models/sim.sp.30k.model"
MAX_SEQ_LENGTH=40
NUM_PROCESSES=16
MAX_VOCAB_SIZE=20000
AE_NLAYERS=1
AE_DROPOUT=0.0
AE_ANNEAL_NOISE_EVERY=500
AE_ANNEAL_NOISE=0.9995
DISC_NUM_STEPS=5
NOISE_R=0.1
GRAD_CLIP_NORM=1.0
GRAD_LAMBDA=0.01
PRECISION=32
SEED=4567
EXP_NAME="[IMDB_CLF_CONTRA_5]"
PYTHON_FILE=${PROJECT_DIR}/"dct/pl_arae_contra_train.py"
WANDB_PROJECT_NAME="DCT_CLF_CONTRA"
NUM_NEAREST_LABELS=2


python ${PYTHON_FILE} \
--exp_name ${EXP_NAME} \
--exp_dir "${EXP_DIR}/${EXP_NAME}" \
--src_train_file "${DATA_DIR}/imdb/train.neg" \
--trg_train_file "${DATA_DIR}/imdb/train.pos" \
--src_dev_file "${DATA_DIR}/imdb/dev.neg" \
--trg_dev_file "${DATA_DIR}/imdb/dev.pos" \
--src_test_file "${DATA_DIR}/imdb/test.neg" \
--trg_test_file "${DATA_DIR}/imdb/test.pos" \
--src-train-label-file "${DATA_DIR}/imdb/train.neg.labels" \
--trg-train-label-file "${DATA_DIR}/imdb/train.pos.labels" \
--src-dev-label-file "${DATA_DIR}/imdb/dev.neg.labels" \
--trg-dev-label-file "${DATA_DIR}/imdb/dev.pos.labels" \
--src-test-label-file "${DATA_DIR}/imdb/test.neg.labels" \
--trg-test-label-file "${DATA_DIR}/imdb/test.pos.labels" \
--src-saliency-file "${DATA_DIR}/imdb/src_dom.attributes.txt" \
--trg-saliency-file "${DATA_DIR}/imdb/trg_dom.attributes.txt" \
--max_seq_length ${MAX_SEQ_LENGTH} \
--batch_size ${BATCH_SIZE} \
--num_processes ${NUM_PROCESSES} \
--limit_train_proportion ${TRAIN_PROPORTION} \
--limit_dev_proportion ${TRAIN_PROPORTION} \
--max_vocab_size ${MAX_VOCAB_SIZE} \
--gpu ${GPUS[0]} \
--ae_emb_size ${HIDDEN_SIZE} \
--ae_hidden_size ${HIDDEN_SIZE} \
--ae_n_layers ${AE_NLAYERS} \
--noise_r ${NOISE_R} \
--ae_dropout ${AE_DROPOUT} \
--lr_ae ${LR_AE} \
--lr-disc ${LR_DISC} \
--lr-gen ${LR_GEN} \
--disc_num_steps ${DISC_NUM_STEPS} \
--ae_anneal_noise_every ${AE_ANNEAL_NOISE_EVERY} \
--ae_anneal_noise ${AE_ANNEAL_NOISE} \
--n_epochs ${N_EPOCHS} \
--grad_clip_norm ${GRAD_CLIP_NORM} \
--arch_d "${HIDDEN_SIZE}-${HIDDEN_SIZE}" \
--arch_g "${HIDDEN_SIZE}-${HIDDEN_SIZE}" \
--gan_gp_lambda ${GAN_GP_LAMBDA} \
--grad_lambda ${GRAD_LAMBDA} \
--seed ${SEED} \
--no-early-stopping \
--generate-through-trg \
--precision ${PRECISION} \
--single-encoder-two-decoders \
--gen-greedy \
--ft-model-path ${FT_MODEL} \
--cola-roberta-checkpoints-dir ${ROBERTA_BASE_CHECKPOINTS_DIR} \
--cola-roberta-json-file ${ROBERTA_BASE_HPARAMS_FILE} \
--sim-model ${SIM_MODEL} \
--sim-sentencepiece-model ${SIM_SENTENCEPIECE_MODEL} \
--wandb-proj-name ${WANDB_PROJECT_NAME} \
--num-nearest-labels ${NUM_NEAREST_LABELS} \
--weight-contrastive-loss 0.7 \
--use-disc-attr-clf \
--arch-contra-clf-baseline "${HIDDEN_SIZE}-${HIDDEN_SIZE}"