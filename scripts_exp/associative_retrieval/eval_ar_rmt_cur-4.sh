#!/usr/bin/env bash

cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_rmt.language_modeling:MemoryCell
RECURRENT_WRAPPER=modeling_rmt.language_modeling:RecurrentWrapper
BACKBONE_CLS=base_models.modeling_gpt_neox:GPTNeoXForCausalLM
TASK_NAME=associative_retrieval
METRIC=exact_match


MODEL_NAME=gpt-neox
MEMORY_SIZE=4

BS=4096
TBS=$((BS*NP))
INPUT_SIZE=2048

# NUMS_PAIRS=(50 100 200 1 2 3 5 10 20 40 300 400 500 700 1000)
NUMS_PAIRS=(400 500 700 1000)


DIM=128
NUM_LAYERS=4


for N in 3
do

for (( j=0; j<${#NUMS_PAIRS[@]}; j++ ))
do
NUM_PAIRS=${NUMS_PAIRS[j]}
KEY_SIZE=3
VALUE_SIZE=1
MAX_N_SEGMENTS=$((NUM_PAIRS + 1))
ITERS=100


# PREV_KEY_SIZES=(1 1 1 1 1 2 2 2)
# PREV_NUMS_PAIRS=(1 2 3 5 10 20 40 50)
PREV_KEY_SIZES=(3)
PREV_NUMS_PAIRS=(50)
for (( k=0; k<${#PREV_NUMS_PAIRS[@]}; k++ ))
do
PREV_KEY_SIZE=${PREV_KEY_SIZES[k]}
PREV_NUM_PAIRS=${PREV_NUMS_PAIRS[k]}
PREV_MAX_N_SEGMENTS=$((PREV_NUM_PAIRS + 1))

BLOCK_SIZE=$((KEY_SIZE + VALUE_SIZE + 2))
cd base_models/gptconfigs
python create_config.py --hidden_size $DIM --num_hidden_layers $NUM_LAYERS --num_attention_heads $NUM_LAYERS
cd ../..
MODEL_CFG=/home/jovyan/rmt/wip/base_models/gptconfigs/neox_tiny_${NUM_LAYERS}l${NUM_LAYERS}hd${DIM}.json

for LR in 3e-04
do

K2=${MAX_N_SEGMENTS}

for SEGMENT_ORDERING in regular
do

for SCHEDULER in linear
do

MODEL_CPT=../runs/test/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_k${PREV_KEY_SIZE}-v1-p${PREV_NUM_PAIRS}-${PREV_MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_${SEGMENT_ORDERING}_bptt-${PREV_MAX_N_SEGMENTS}_${NUM_LAYERS}l${NUM_LAYERS}hd${DIM}/run_$N 

GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
NP=$NP
ACCEL_CONFIG=/home/jovyan/rmt/babilong/accel_configs/deepspeed_bf16_tbs${TBS}bs${BS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml
cd accel_configs/
python create_config.py \
        --bf16 \
        --train_batch_size $TBS\
        --train_micro_batch_size_per_gpu $BS\
        --gradient_accumulation_steps $GRAD_ACC_STEPS\
        --np $NP\
        --gradient_clipping 1.0
cd ..

echo gradient accumulation steps $GRAD_ACC_STEPS

echo RUNNING: TASK_NAME MEMORY_SIZE KEY_SIZE VALUE_SIZE N_SEG  MODEL_NAME MODEL_CLS LR N
echo RUNNING: $TASK_NAME $MEMORY_SIZE $KEY_SIZE $VALUE_SIZE $MAX_N_SEGMENTS $MODEL_NAME $MODEL_CLS  $LR $N
accelerate launch --config_file $ACCEL_CONFIG --main_process_port 29571 run_finetuning_associative_retrieval.py \
        --task_name $TASK_NAME \
        --model_path ../runs/test/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_k${KEY_SIZE}-v${VALUE_SIZE}-p${NUM_PAIRS}-${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs512_${SEGMENT_ORDERING}_bptt-${K2}_${NUM_LAYERS}l${NUM_LAYERS}hd${DIM}_eval_from_p${PREV_NUM_PAIRS}/run_$N \
        --model_cfg $MODEL_CFG \
        --model_cls $BACKBONE_CLS \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --segment_size $BLOCK_SIZE \
        --key_size $KEY_SIZE \
        --value_size $VALUE_SIZE \
        --num_pairs $NUM_PAIRS \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --use_generate_on_valid \
        --batch_size $BS --gradient_accumulation_steps $GRAD_ACC_STEPS \
        --iters $ITERS \
        --num_training_steps $((ITERS*2)) \
        --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval 100 --valid_interval 500 \
        --optimize_metric $METRIC --optimize_mode max \
        --show_valid_examples 5 \
        --early_stopping_patience 50 \
        --seed $(($N+42)) \
        --clip_grad_value 1.0 \
        --train_size 1000000 \
        --valid_size 1000 \
        --test_size 10000 \
        --model_cpt ../runs/test/associative_retrieval/gpt-neox/lr3e-04_linear_adamw_wd1e-03_k3-v1-p50-51x2048_mem4_bs512_regular_bptt-51_4l4hd128_continue/run_3/ \
        --validate_only
done
done
done
done
done
done
echo "done"