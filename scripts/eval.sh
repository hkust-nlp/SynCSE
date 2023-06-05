root=/data/users/zhangjunlei/code/simcse_official/SimCSE-main/result/
path=${root}/my-sup-simcse-roberta-base_0528_nli_241789_241789
#path=my-sup-simcse-roberta-base_final_nli_0528_13/
python simcse_to_huggingface.py --path ${path}
CUDA_VISIBLE_DEVICES=0 python evaluation.py \
    --model_name_or_path ${path} \
    --pooler cls \
    --task_set sts \
    --mode test
