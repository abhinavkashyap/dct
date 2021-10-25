#!/usr/bin/env bash
PROJECT_DIR="/home/ubuntu/abhi/dct"
DUMP_DIR="/abhinav/dct/"
DATA_DIR=${DUMP_DIR}/"data"
EXP_DIR=${DUMP_DIR}/"experiments"
PYTHON_FILE=${PROJECT_DIR}/"dct/train_dann.py"
MAX_SEQ_LENGTH=100
BATCH_SIZE=32
EXP_NAME="[DANN_DVD_ELECTRONICS]"
ENC_EMB_SIZE=300
GLOVE_NAME="6B"
GLOVE_DIM=300
MAX_VOCAB_SIZE=3000
ENC_HIDDEN_SIZE=512
NUM_ENC_LAYERS=1
LINEAR_CLF_HIDDEN_DIM=512
DOMAIN_DISC_OUT_DIM=2
TASK_CLF_OUT_DIM=2
DANN_ALPHAS=(0.07 0.08 0.09)
TRAIN_PROPORTION=1.0
DEV_PROPORTION=1.0
TEST_PROPORTION=1.0
SEED=1729
LR=1e-3
EPOCHS=20
GPU=0
GRAD_CLIP_NORM=5.0
WANDB_PROJ_NAME="DANN"
WEIGHT_DOM_LOSS=1.0


for dann_alpha in ${DANN_ALPHAS[@]};
do
    python ${PYTHON_FILE} \
    --exp_name ${EXP_NAME}_${dann_alpha}alpha \
    --exp_dir ${EXP_DIR}/${EXP_NAME}_${dann_alpha}alpha \
    --src_dom_train_filename ${DATA_DIR}/"mcauley_reviews/dvd.dann.train" \
    --src_dom_dev_filename ${DATA_DIR}/"mcauley_reviews/dvd.dann.dev" \
    --src_dom_test_filename ${DATA_DIR}/"mcauley_reviews/dvd.dann.test" \
    --trg_dom_train_filename ${DATA_DIR}/"mcauley_reviews/electronics.dann.train" \
    --trg_dom_dev_filename ${DATA_DIR}/"mcauley_reviews/electronics.dann.dev" \
    --trg_dom_test_filename ${DATA_DIR}/"mcauley_reviews/electronics.dann.test" \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --batch_size ${BATCH_SIZE} \
    --enc_emb_size ${ENC_EMB_SIZE} \
    --enc_hidden_size ${ENC_HIDDEN_SIZE} \
    --num_enc_layers ${NUM_ENC_LAYERS} \
    --domain_disc_out_dim ${DOMAIN_DISC_OUT_DIM} \
    --task_clf_out_dim ${TASK_CLF_OUT_DIM} \
    --dann_alpha ${dann_alpha} \
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
    --linear_clf_hidden_dim ${LINEAR_CLF_HIDDEN_DIM} \
    --use_gru_encoder \
    --glove_name ${GLOVE_NAME} \
    --glove_dim ${GLOVE_DIM}
done