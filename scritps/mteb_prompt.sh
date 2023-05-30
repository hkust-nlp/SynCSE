#!/bin/bash
EXP=sup-roberta
model_path=/data/users/zhangjunlei/download/sup-roberta


case "$EXP" in

"unsup-roberta")
    CUDA_VISIBLE_DEVICES=1 python eval_prompt_mteb.py \
        --model_name_or_path  ${model_path}\
        --mask_embedding_sentence \
        --mask_embedding_sentence_template "*cls*_This_sentence_:_'_*sent_0*_'_means*mask*.*sep+*"
    ;;

"sup-roberta")
    CUDA_VISIBLE_DEVICES=3 python eval_prompt_mteb.py \
        --model_name_or_path  ${model_path}\
        --mask_embedding_sentence \
        --mask_embedding_sentence_use_pooler\
        --mask_embedding_sentence_delta \
        --mask_embedding_sentence_template "*cls*_This_sentence_:_'_*sent_0*_'_means*mask*.*sep+*"
    ;;
*)
esac