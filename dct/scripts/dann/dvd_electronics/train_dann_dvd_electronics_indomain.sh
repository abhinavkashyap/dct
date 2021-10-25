#!/usr/bin/env bash
PROJECT_DIR="/home/rkashyap/abhi/dct"
DUMP_DIR="/data3/abhinav/dct/"
DATA_DIR=${DUMP_DIR}/"data"
EXP_DIR=${DUMP_DIR}/"experiments"
PYTHON_FILE=${PROJECT_DIR}/"dct/train_dann.py"
MAX_SEQ_LENGTH=100
BATCH_SIZE=32
EXP_NAME="[DANN_DVD_ELECTRONICS_INDOMAIN]"
ENC_EMB_SIZE=300
GLOVE_NAME="6B"
GLOVE_DIM=300
MAX_VOCAB_SIZE=3000
ENC_HIDDEN_SIZE=512
NUM_ENC_LAYERS=1
LINEAR_CLF_HIDDEN_DIM=512
DOMAIN_DISC_OUT_DIM=2
TASK_CLF_OUT_DIM=2
DANN_ALPHA=0.1
TRAIN_PROPORTION=1.0
DEV_PROPORTION=1.0
TEST_PROPORTION=1.0
SEED=1729
LR=1e-3
EPOCHS=20
GPU=0
GRAD_CLIP_NORM=5.0
WANDB_PROJ_NAME="DANN"
WEIGHT_DOM_LOSS=0.0



python ${PYTHON_FILE} \
--exp_name ${EXP_NAME} \
--exp_dir ${EXP_DIR}/${EXP_NAME} \
--src_dom_train_filename ${DATA_DIR}/"processed_blitzer_reviews/dvd/dvd.train" \
--src_dom_dev_filename ${DATA_DIR}/"processed_blitzer_reviews/dvd/dvd.dev" \
--src_dom_test_filename ${DATA_DIR}/"processed_blitzer_reviews/dvd/dvd.test" \
--trg_dom_train_filename ${DATA_DIR}/"processed_blitzer_reviews/electronics/electronics.train" \
--trg_dom_dev_filename ${DATA_DIR}/"processed_blitzer_reviews/electronics/electronics.dev" \
--trg_dom_test_filename ${DATA_DIR}/"processed_blitzer_reviews/electronics/electronics.test" \
--max_seq_length ${MAX_SEQ_LENGTH} \
--batch_size ${BATCH_SIZE} \
--enc_emb_size ${ENC_EMB_SIZE} \
--enc_hidden_size ${ENC_HIDDEN_SIZE} \
--num_enc_layers ${NUM_ENC_LAYERS} \
--domain_disc_out_dim ${DOMAIN_DISC_OUT_DIM} \
--task_clf_out_dim ${TASK_CLF_OUT_DIM} \
--dann_alpha ${DANN_ALPHA} \
--weight_dom_loss ${WEIGHT_DOM_LOSS} \
--train_proportion ${TRAIN_PROPORTION} \
--dev_proportion ${DEV_PROPORTION} \
--test_proportion ${TEST_PROPORTION} \
--seed ${SEED} \
--lr ${LR} \
--epochs ${EPOCHS} \
--gpu ${GPU} \
--grad_clip_norm ${GRAD_CLIP_NORM} \
--wandb_proj_name ${WANDB_PROJ_NAME} \
--no_adv_train \
--linear_clf_hidden_dim ${LINEAR_CLF_HIDDEN_DIM} \
--use_gru_encoder \
--glove_name ${GLOVE_NAME} \
--glove_dim ${GLOVE_DIM}