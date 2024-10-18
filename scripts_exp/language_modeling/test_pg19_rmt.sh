#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./finetune_babilong_baseline.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_rmt.language_modeling:MemoryCell
RECURRENT_WRAPPER=modeling_rmt.language_modeling:RecurrentWrapper
BACKBONE_CLS=transformers:AutoModelForCausalLM
DATASET_NAME=pg19
METRIC=exact_match

MODEL_NAME=llama32-1b  # backbone model
MODEL_PATH=/home/jovyan/kuratov/models/Llama-3.2-1B

TASK_DATASET=/home/jovyan/rmt/datasets/pg19/pg19_tokenized

ITERS=50000
TBS=32
BS=4

for LR in 1e-04
do

for SEGMENT_SIZE in 1024 # size of one segment in tokens
do

for MAX_N_SEGMENTS in 2
do

for MEMORY_SIZE in 16
do

SAMPLE_SIZE=$((MAX_N_SEGMENTS*SEGMENT_SIZE)) # length of task sample in tokens
GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
SCHEDULER=linear

for N in 1
do

K2=-1   # BPTT unroll length

NP=$NP
# ACCEL_CONFIG=/home/jovyan/rmt/dev/accel_configs/accelerate_bf16.yaml
ACCEL_CONFIG=/home/jovyan/rmt/dev/accel_configs/accelerate_ds_bf16.yaml
DEEPSPEED_CONFIG=/home/jovyan/rmt/dev/accel_configs/deepspeed_bf16_test.json

echo RUNNING: DATASET_NAME $DATASET_NAME MEMORY_SIZE $MEMORY_SIZE SEGMENT_SIZE $SEGMENT_SIZE MAX_N_SEGMENTS $MAX_N_SEGMENTS
echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME  LR $LR N $N
echo gradient accumulation steps $GRAD_ACC_STEPS

# python run_finetuning_lm_rmt.py \
accelerate launch --config_file $ACCEL_CONFIG --deepspeed_config_file $DEEPSPEED_CONFIG --main_process_port 29000 run_finetuning_lm_rmt.py \
        --tokenized_dataset $TASK_DATASET \
        --output_dir /home/jovyan/rmt/runs/test/${DATASET_NAME}/$MODEL_NAME/${SCHEDULER}_adamw_wd1e-03_${MAX_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_bptt-${K2}-test/run_$N \
        --from_pretrained $MODEL_PATH \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --segment_size $SEGMENT_SIZE \
        --sample_size $SAMPLE_SIZE \
        --val_sample_size $SAMPLE_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --per_device_train_batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --max_steps $ITERS \
        --metric_for_best_model "eval_loss" \
        --greater_is_better false \
        --save_total_limit 1 \
        --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.01 \
        --learning_rate ${LR} --lr_scheduler_type $SCHEDULER --warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --logging_steps 25 --eval_steps 100 \
        --show_valid_examples 5 \
        --seed $(($N+42)) \
        
done
done
done
done
done
done
done
done
echo "done"
