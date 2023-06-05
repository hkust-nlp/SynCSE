#!/bin/bash

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=1

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
PORT_ID=$(expr $RANDOM + 1000)

# Allow multiple threads
export OMP_NUM_THREADS=1

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \

model=roberta-base
dataset=sjtu-lit/SynCSE-partial-NLI
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --model_name_or_path ${model} \
    --train_file  ${dataset} \
    --output_dir result/my-sup-simcse-${model}_${dataset} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 512 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model avg_sts \
    --load_best_model_at_end \
    --eval_steps 25 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16 \
    --seed 42 \
    --do_mlm \
    --hard_negative_weight 0 \
    "$@"
