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

NUMS_PAIRS=(2 3 5 10 20 40 50 50 200)
KEY_SIZES=(1 1 1 1 2 2 2 3 3)
VALUE_SIZES=(1 1 1 1 1 1 1 1 1)
BSS=(256 256 256 256 256 256 256 256 64)
ITERSS=(50000 50000 50000 50000 50000 50000 50000 50000 50000)
LRS=(3e-04 3e-04 3e-04 3e-04 1e-04 1e-04 1e-04 1e-04 1e-04)
#LRS=(3e-04 3e-04 3e-04 3e-04 3e-04 3e-04 3e-04 3e-04 3e-04)

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
LR=${LRS[j]}

BLOCK_SIZE=$((KEY_SIZE + VALUE_SIZE + 2))
cd base_models/gptconfigs
python create_config.py --hidden_size $DIM --num_hidden_layers $NUM_LAYERS --num_attention_heads $NUM_LAYERS
cd ../..
MODEL_CFG=./base_models/gptconfigs/neox_tiny_${NUM_LAYERS}l${NUM_LAYERS}hd${DIM}.json

for x_READ_MEM in 2
do
READ_MEM_SIZE=$((MEMORY_SIZE*x_READ_MEM))
WRITE_MEM_SIZE=$MEMORY_SIZE

K2=-1

for SEGMENT_ORDERING in regular
do

for SCHEDULER in linear
do

if [[ j -gt 0 ]]
then
    PREV_NUM_PAIRS=${NUMS_PAIRS[j-1]}
    PREV_MAX_N_SEGMENTS=$((PREV_NUM_PAIRS + 1))
    PREV_LR=${LRS[j-1]}
    MODEL_CPT=./runs/${TASK_NAME}/$MODEL_NAME/lr${PREV_LR}_${SCHEDULER}_adamw_wd1e-03_k${KEY_SIZES[j-1]}-v${VALUE_SIZES[j-1]}-p${PREV_NUM_PAIRS}-${PREV_MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_r${READ_MEM_SIZE}_w${WRITE_MEM_SIZE}_retr_${RETR_MODE}_bs${TBS}_${SEGMENT_ORDERING}_bptt-${K2}_${NUM_LAYERS}l${NUM_LAYERS}hd${DIM}/run_$N 
else
    MODEL_CPT=None
fi

GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
ACCEL_CONFIG=./accel_configs/accelerate.yaml

echo gradient accumulation steps $GRAD_ACC_STEPS

echo RUNNING: TASK_NAME MEMORY_SIZE KEY_SIZE VALUE_SIZE N_SEG  MODEL_NAME MODEL_CLS LR N
echo RUNNING: $TASK_NAME $MEMORY_SIZE $KEY_SIZE $VALUE_SIZE $MAX_N_SEGMENTS $MODEL_NAME $MODEL_CLS  $LR $N
# accelerate launch --config_file $ACCEL_CONFIG --main_process_port 29571 run_finetuning_associative_retrieval_repeat_question.py \
python run_finetuning_associative_retrieval_repeat_question.py \
        --task_name $TASK_NAME \
        --dataset_path /home/jovyan/kuratov/rmt/datasets/associative_retrieval_rq \
        --model_path ./runs/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_k${KEY_SIZE}-v${VALUE_SIZE}-p${NUM_PAIRS}-${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_r${READ_MEM_SIZE}_w${WRITE_MEM_SIZE}_retr_${RETR_MODE}_bs${TBS}_${SEGMENT_ORDERING}_bptt-${K2}_${NUM_LAYERS}l${NUM_LAYERS}hd${DIM}/run_$N \
        --model_cpt ${MODEL_CPT} \
        --model_cfg $MODEL_CFG \
        --model_cls $BACKBONE_CLS \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --segment_size $BLOCK_SIZE \
        --key_size $KEY_SIZE \
        --value_size $VALUE_SIZE \
        --num_pairs $NUM_PAIRS \
        --num_read_mem_tokens $READ_MEM_SIZE \
        --num_write_mem_tokens $WRITE_MEM_SIZE \
        --retrieve_mode $RETR_MODE \
        --max_n_segments $MAX_N_SEGMENTS\
        --use_generate_on_valid \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --iters $ITERS \
        --num_training_steps $((ITERS*2)) \
        --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.000 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 1 \
        --log_interval 200 --valid_interval 200 \
        --optimize_metric $METRIC --optimize_mode max \
        --show_valid_examples 5 \
        --early_stopping_patience 25 \
        --seed $(($N+42)) \
        --train_size 1000000 \
        --valid_size 1000 \
        --test_size 10000 \
        --save_best
done
done
done
done
done
echo "done"