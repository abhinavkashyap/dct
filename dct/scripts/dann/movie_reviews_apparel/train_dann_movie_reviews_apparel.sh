#!/usr/bin/env bash
PROJECT_DIR="/home/rkashyap/abhi/dct"
DUMP_DIR="/data3/abhinav/dct/"
DATA_DIR=${DUMP_DIR}/"data"
EXP_DIR=${DUMP_DIR}/"experiments"
PYTHON_FILE=${PROJECT_DIR}/"dct/train_dann.py"
MAX_SEQ_LENGTH=50
BATCH_SIZE=16
ENC_EMB_SIZE=200
GLOVE_NAME="6B"
GLOVE_DIM=300
MAX_VOCAB_SIZE=10000
ENC_HIDDEN_SIZE=128
NUM_ENC_LAYERS=1
LINEAR_CLF_HIDDEN_DIM=128
DOMAIN_DISC_OUT_DIM=2
TASK_CLF_OUT_DIM=2
DANN_ALPHAS=(0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09)
TRAIN_PROPORTION=1.0
DEV_PROPORTION=1.0
TEST_PROPORTION=1.0
SEED=1729
LR=1e-3
EPOCHS=50
GPU=0
GRAD_CLIP_NORM=2.0
WANDB_PROJ_NAME="DANN"
WEIGHT_DOM_LOSS=1.0


for DANN_ALPHA in ${DANN_ALPHAS[@]};
do
    EXP_NAME="[DANN_MR_APPAREL_alpha${DANN_ALPHA}]"
    python ${PYTHON_FILE} \
    --exp_name ${EXP_NAME} \
    --exp_dir ${EXP_DIR}/${EXP_NAME} \
    --src_dom_train_filename ${DATA_DIR}/"sa/MR.task.train" \
    --src_dom_dev_filename ${DATA_DIR}/"sa/MR.task.dev" \
    --src_dom_test_filename ${DATA_DIR}/"sa/MR.task.test" \
    --trg_dom_train_filename ${DATA_DIR}/"sa/apparel.task.train" \
    --trg_dom_dev_filename ${DATA_DIR}/"sa/apparel.task.dev" \
    --trg_dom_test_filename ${DATA_DIR}/"sa/apparel.task.test" \
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
    --linear_clf_hidden_dim ${LINEAR_CLF_HIDDEN_DIM} \
    --use_gru_encoder
done