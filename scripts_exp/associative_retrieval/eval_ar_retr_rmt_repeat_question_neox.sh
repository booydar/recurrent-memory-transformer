#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_rmt.language_modeling_mem_retrieve:MemoryCell
RECURRENT_WRAPPER=modeling_rmt.language_modeling_mem_retrieve:RecurrentWrapper
BACKBONE_CLS=base_models.modeling_gpt_neox:GPTNeoXForCausalLM
TASK_NAME=associative_retrieval_rq
METRIC=exact_match

MODEL_NAME=gpt-neox
MEMORY_SIZE=4
RETR_MODE='attention'

TBS=256
INPUT_SIZE=2048
BS=256

NUMS_PAIRS=(2 3 5 10 20 40 50 50 100 200 300 400 500 700 1000)
KEY_SIZES=(1 1 1 1 2 2 2 3 3 3 3 3 3 3 3)
VALUE_SIZES=(1 1 1 1 1 1 1 1 1 1 1 1 1 1 1)

DIM=128
NUM_LAYERS=4

for (( j=0; j<${#NUMS_PAIRS[@]}; j++ ))
do
NUM_PAIRS=${NUMS_PAIRS[j]}
KEY_SIZE=${KEY_SIZES[j]}
VALUE_SIZE=${VALUE_SIZES[j]}
MAX_N_SEGMENTS=$((NUM_PAIRS + 1))

BLOCK_SIZE=$((KEY_SIZE + VALUE_SIZE + 2))

GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
ACCEL_CONFIG=./accel_configs/accelerate.yaml

RUNS_PATH=./runs/${TASK_NAME}/${MODEL_NAME}
EXP_PATH=${RUNS_PATH}/lr1e-04_linear_adamw_wd1e-03_k3-v1-p50-51x2048_mem4_r8_w4_retr_attention_bs256_regular_bptt--1_4l4hd128/run_5


echo gradient accumulation steps $GRAD_ACC_STEPS

echo RUNNING: TASK_NAME MEMORY_SIZE KEY_SIZE VALUE_SIZE N_SEG  MODEL_NAME MODEL_CLS LR N
echo RUNNING: $TASK_NAME $MEMORY_SIZE $KEY_SIZE $VALUE_SIZE $MAX_N_SEGMENTS $MODEL_NAME $MODEL_CLS  $LR $N
# accelerate launch --config_file $ACCEL_CONFIG --main_process_port 29571 run_finetuning_associative_retrieval_repeat_question.py \
python run_finetuning_associative_retrieval_repeat_question.py \
        --task_name $TASK_NAME \
        --dataset_path /home/jovyan/kuratov/rmt/datasets/associative_retrieval_rq \
        --experiment_cfg ${EXP_PATH}/config.json \
        --init_checkpoint ${EXP_PATH}/model_best/pytorch_model.bin \
        --segment_size $BLOCK_SIZE \
        --key_size $KEY_SIZE \
        --value_size $VALUE_SIZE \
        --num_pairs $NUM_PAIRS \
        --max_n_segments $MAX_N_SEGMENTS\
        --use_generate_on_valid \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --data_n_workers 1 \
        --show_valid_examples 5 \
        --validate_only
done
echo "done"