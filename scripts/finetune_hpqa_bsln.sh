#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=5,7,9 NP=3 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=encoder
MODEL_NAME=roberta-base
MODEL_CLS=modeling_roberta_bsln:RobertaHotpotQA
TASK_NAME=distractor
METRIC=exact_match

ITERS=3000
TBS=6   
BS=1

TGT_LEN=512
INPUT_SEQ_LEN=32



SCHEDULER=linear

for LR in 1e-05
do


echo RUNNING: TASK_NAME SRC_LEN MODEL_NAMEcuda INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N
horovodrun --gloo -np $NP python run_finetuning_hotpot.py \
        --task_name $TASK_NAME \
        --model_path ../runs/debug/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}-_bs${TBS}_iters${ITERS}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --input_seq_len $INPUT_SEQ_LEN \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --iters $ITERS \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval $(($ITERS/100)) --valid_interval $(($ITERS/10)) \
        --optimize_metric $METRIC --optimize_mode max \
        --early_stopping_patience 15 \
        --optimize_metric $METRIC --optimize_mode max \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
done
done
done
done
echo "run_bert_pretraining.py done"
echo "done"
