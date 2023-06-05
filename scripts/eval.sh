path=sjtu-lit/SynCSE-partial-RoBERTa-base
python simcse_to_huggingface.py --path ${path}
CUDA_VISIBLE_DEVICES=0 python evaluation.py \
    --model_name_or_path ${path} \
    --pooler cls \
    --task_set sts \
    --mode test
