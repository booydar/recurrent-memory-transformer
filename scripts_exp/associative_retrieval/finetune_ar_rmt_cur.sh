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

TBS=512
INPUT_SIZE=2048

NUMS_PAIRS=(1 2 3 5 10 20 40 50)
KEY_SIZES=(1 1 1 1 1 2 2 2)
VALUE_SIZES=(1 1 1 1 1 1 1 1)
BSS=(128 128 128 128 128 128 128 128)
ITERSS=(2000 10000 10000 10000 10000 10000 10000 30000)

# ITERSS=(1 1 1 1 1 1 1 1)

DIM=128
NUM_LAYERS=4


for N in 1
do

for (( j=0; j<${#NUMS_PAIRS[@]}; j++ ))
do
NUM_PAIRS=${NUMS_PAIRS[j]}
KEY_SIZE=${KEY_SIZES[j]}
VALUE_SIZE=${VALUE_SIZES[j]}
MAX_N_SEGMENTS=$((NUM_PAIRS + 1))
BS=${BSS[j]}
ITERS=${ITERSS[j]}


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

if [[ j -gt 0 ]]
then
    PREV_NUM_PAIRS=${NUMS_PAIRS[j-1]}
    PREV_MAX_N_SEGMENTS=$((PREV_NUM_PAIRS + 1))
    MODEL_CPT=../runs/${TASK_NAME}/rmt/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_k${KEY_SIZES[j-1]}-v${VALUE_SIZES[j-1]}-p${PREV_NUM_PAIRS}-${PREV_MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_${SEGMENT_ORDERING}_bptt-${PREV_MAX_N_SEGMENTS}_${NUM_LAYERS}l${NUM_LAYERS}hd${DIM}/run_$N 
else
    MODEL_CPT=None
fi

GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
ACCEL_CONFIG=./accel_configs/accelerate.yaml

echo gradient accumulation steps $GRAD_ACC_STEPS

echo RUNNING: TASK_NAME MEMORY_SIZE KEY_SIZE VALUE_SIZE N_SEG  MODEL_NAME MODEL_CLS LR N
echo RUNNING: $TASK_NAME $MEMORY_SIZE $KEY_SIZE $VALUE_SIZE $MAX_N_SEGMENTS $MODEL_NAME $MODEL_CLS  $LR $N
# accelerate launch --config_file $ACCEL_CONFIG --main_process_port 29571 run_finetuning_associative_retrieval.py \
python run_finetuning_associative_retrieval.py \
        --task_name $TASK_NAME \
        --model_path ../runs/test/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_k${KEY_SIZE}-v${VALUE_SIZE}-p${NUM_PAIRS}-${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_${SEGMENT_ORDERING}_bptt-${K2}_${NUM_LAYERS}l${NUM_LAYERS}hd${DIM}/run_$N \
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
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
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
        --vary_n_segments \
        --model_cpt $MODEL_CPT \
        --save_best
done
done
done
done
done
echo "done"