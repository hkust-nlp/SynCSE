#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

CUDA_VISIBLE_DEVICES=6 python train.py \
    --model_name_or_path roberta-base \
    --train_file /data/users/zhangjunlei/code/simcse_official/SimCSE-main/data/nli_for_simcse.csv \
    --output_dir result/nli_for_simcse \
    --num_train_epochs 3 \
    --per_device_train_batch_size 512 \
    --learning_rate 5e-5 \
    --max_seq_length 64 \
    --evaluation_strategy steps \
    --metric_for_best_model sts_ \
    --load_best_model_at_end \
    --eval_steps 25 \
    --pooler_type cls_before_pooler \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    "$@"
