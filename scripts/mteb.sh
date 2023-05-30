root=/data/users/zhangjunlei/code/simcse_official/SimCSE-main/result/
#model_path=${root}/my-sup-simcse-roberta-large_combined_data
echo ${model_path}
model_path=qiyuw/pcl-roberta-base
CUDA_VISIBLE_DEVICES=5 python eval_mteb.py --startid 0 \
                                          --endid -1 \
                                          --modelpath ${model_path}
