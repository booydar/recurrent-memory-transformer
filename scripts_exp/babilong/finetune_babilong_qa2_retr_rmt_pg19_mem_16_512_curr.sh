#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./finetune_babilong_baseline.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_rmt.language_modeling_mem_retrieve:MemoryCell
RECURRENT_WRAPPER=modeling_rmt.language_modeling_mem_retrieve:RecurrentWrapper
BACKBONE_CLS=transformers:AutoModelForCausalLM
TASK_DATASET=qa2_two-supporting-facts
NOISE_DATASET=pg19
METRIC=exact_match

MODEL_NAME=gpt2  # backbone model
SEGMENT_SIZE=512 # size of one segment in tokens
SAMPLE_SIZE=2048 # length of task sample in tokens
MEMORY_SIZE=16
MAX_N_SEGMENTS=4

SAMPLING_PROB=0.5

RETR_MODE='attention'

ITERS=10000
TBS=64
BS=1

SCHEDULER=linear
LR=1e-04
WD=1e-03

# -> prepare checkpoint trained on 1 segm
MAX_N_SEGMENTSS=(0 2 4 6 8 16 32)
BSS=(8 8 8 4 4 2 1)

for N in 2
do
for MEMORY_SIZE in 16 # 16 32
do
for LR in 5e-05
do
for (( j=2; j<${#MAX_N_SEGMENTSS[@]}; j++ ))
do
BS=${BSS[j]}
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]}

j1=$((j-1))
SRC_N_SEGMENTS=${MAX_N_SEGMENTSS[j1]}

j2=$((j-2))
SRC_SRC_N_SEGMENTS=${MAX_N_SEGMENTSS[j2]}

for x_READ_MEM in 2
do
READ_MEM_SIZE=$((MEMORY_SIZE*x_READ_MEM))
WRITE_MEM_SIZE=$MEMORY_SIZE

SAMPLE_SIZE=$((MAX_N_SEGMENTS*SEGMENT_SIZE)) # length of task sample in tokens
PREV_SAMPLE_SIZE=$((SRC_N_SEGMENTS*SEGMENT_SIZE))

K2=-1   # BPTT unroll length

ACCEL_CONFIG=./accelerate.yaml
GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))

MODEL_PATH=./runs/babilong/${TASK_DATASET}/${NOISE_DATASET}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd${WD}_seqlen${SAMPLE_SIZE}_${MAX_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_r${READ_MEM_SIZE}_w${WRITE_MEM_SIZE}_bs${TBS}_${SEGMENT_ORDERING}_bptt-${K2}_sp${SAMPLING_PROB}_retr_${RETR_MODE}_from_${SRC_N_SEGMENTS}x${SEGMENT_SIZE}/run_$N
INIT_MODEL_PATH=./runs/babilong/${TASK_DATASET}/${NOISE_DATASET}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd${WD}_seqlen${PREV_SAMPLE_SIZE}_${SRC_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_r${READ_MEM_SIZE}_w${WRITE_MEM_SIZE}_bs${TBS}_${SEGMENT_ORDERING}_bptt-${K2}_sp${SAMPLING_PROB}_retr_${RETR_MODE}_from_${SRC_SRC_N_SEGMENTS}x${SEGMENT_SIZE}/run_$N

echo RUNNING: TASK_DATASET $TASK_DATASET MEMORY_SIZE $MEMORY_SIZE SEGMENT_SIZE $SEGMENT_SIZE 
echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME LR $LR N $N
echo gradient accumulation steps $GRAD_ACC_STEPS
echo $MODEL_PATH
echo from $INIT_MODEL_PATH

accelerate launch --num_processes $NP --main_process_port 29207 --config_file $ACCEL_CONFIG run_finetuning_babilong_rmt.py \
        --task_dataset $TASK_DATASET \
        --noise_dataset $NOISE_DATASET \
        --babi_path /home/jovyan/rmt/datasets/babi/data/tasks_1-20_v1-2/en-10k/ \
        --model_path $MODEL_PATH \
        --init_checkpoint ${INIT_MODEL_PATH}/model_best/pytorch_model.bin \
        --reset_iteration \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --segment_size $SEGMENT_SIZE \
        --sample_size $SAMPLE_SIZE \
        --vary_n_segments --mixed_length_ratio $SAMPLING_PROB \
        --num_read_mem_tokens $READ_MEM_SIZE \
        --num_write_mem_tokens $WRITE_MEM_SIZE \
        --retrieve_mode $RETR_MODE \
        --max_n_segments $MAX_N_SEGMENTS\
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --num_training_steps $((ITERS*2)) \
        --iters $ITERS \
        --save_best \
        --k2 $K2 \
        --optimizer AdamW  --weight_decay $WD \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval 50 --valid_interval 50 \
        --optimize_metric $METRIC --optimize_mode max --save_best \
        --show_valid_examples 1 \
        --early_stopping_patience 15 \
        --seed $(($N+42)) \
        --clip_grad_norm 1.0
done
done
done
done
done
echo "done"
