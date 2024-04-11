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
TASK_DATASET=qa1_single-supporting-fact
NOISE_DATASET=pg19
METRIC=exact_match

MODEL_NAME=gpt2  # backbone model
SEGMENT_SIZE=512 # size of one segment in tokens
SAMPLE_SIZE=512 # length of task sample in tokens
MEMORY_SIZE=16
MAX_N_SEGMENTS=1

SAMPLING_PROB=0.5

RETR_MODE='attention'

ITERS=10000
TBS=64
BS=16

GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
SCHEDULER=linear
LR=1e-04
WD=1e-03

for LR in 5e-05
do
for N in 1
do
for MEMORY_SIZE in 16
do
for x_READ_MEM in 2
do
READ_MEM_SIZE=$((MEMORY_SIZE*x_READ_MEM))
WRITE_MEM_SIZE=$MEMORY_SIZE

K2=-1   # BPTT unroll length

ACCEL_CONFIG=./accelerate.yaml

echo RUNNING: TASK_DATASET $TASK_DATASET MEMORY_SIZE $MEMORY_SIZE SEGMENT_SIZE $SEGMENT_SIZE 
echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME LR $LR N $N
echo gradient accumulation steps $GRAD_ACC_STEPS

TOKENIZERS_PARALLELISM=false
accelerate launch --num_processes $NP --main_process_port 29115 --config_file $ACCEL_CONFIG run_finetuning_babilong_rmt.py \
        --task_dataset $TASK_DATASET \
        --noise_dataset $NOISE_DATASET \
        --babi_path /home/jovyan/rmt/datasets/babi/data/tasks_1-20_v1-2/en-10k/ \
	--model_path ./runs/babilong/${TASK_DATASET}/${NOISE_DATASET}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd${WD}_seqlen${SAMPLE_SIZE}_${MAX_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_r${READ_MEM_SIZE}_w${WRITE_MEM_SIZE}_bs${TBS}_${SEGMENT_ORDERING}_bptt-${K2}_sp${SAMPLING_PROB}_retr_${RETR_MODE}_repeat_last_from_0x${SEGMENT_SIZE}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --segment_size $SEGMENT_SIZE \
        --sample_size $SAMPLE_SIZE \
        --vary_n_segments --mixed_length_ratio $SAMPLING_PROB \
        --repeat_last \
        --num_read_mem_tokens $READ_MEM_SIZE \
        --num_write_mem_tokens $WRITE_MEM_SIZE \
        --retrieve_mode $RETR_MODE \
        --max_n_segments $MAX_N_SEGMENTS \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --num_training_steps $((ITERS+2000)) \
        --iters $ITERS \
        --save_best \
        --k2 $K2 \
        --optimizer AdamW  --weight_decay $WD \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval 100 --valid_interval 100 \
        --optimize_metric $METRIC --optimize_mode max --save_best \
        --show_valid_examples 5 \
        --early_stopping_patience 40 \
        --seed $(($N+42)) \
        --clip_grad_norm 1.0
done
done
done
done
echo "done"

