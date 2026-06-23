#!/bin/bash

for kfold in {1..1}
do
    version1_replaced=$(echo "241014_6_baseline_learnT_swinT_div2_p1_con1_k1_seed20000912" | sed "s/k1/k$kfold/g")
    echo "$version1_replaced"
    export CUDA_VISIBLE_DEVICES=1
    python run.py \
        -t -e \
        --num_layers_PG 11 \
        --num_layers_R 4 \
        --num_R 3 \
        --seed 20000912 \
        --input_dim 768 \
        --mask \
        --pos_w "pre_w" \
        --loss_type "none" \
        --decay_rate 0.99 \
        --dataset_variant cholect45-crossval \
        --kfold $kfold \
        --topk 3 \
        --epochs 800 \
        --batch 1 \
        -l 1e-2 5e-3 5e-3 \
        -w 9 18 400 \
        --version "time_ours"$kfold \
        --version1 "$version1_replaced" \
        --gpu 0 \
        --reverse \
        --val_interval 20 \
        --random 0.5 \
        --alpha 0.1 \
        --winsize 19 \
        -sf 2 \
        --arch 6 5 5 \
        --fpn p1 \
        --model actionformer \
        --clip_t_feature "clip_features_ViT-L-14_feats_32_text.pkl" \
        --clip_i_feature "clip_features_ViT-L-14_feats_32.pkl" \
        --fuse 'none' \
        --clip_loss cos \
        --fusion 'none' \
        --beta 0.1 \
        --power 0.1 \
        --transit 0.8 \
        --gamma 2 \
        --eql \
        --ins_prompt 1 \
        --ins_prompt_source gt_attribute \
        --topk 3 \
        --target_prompt -1 \
        --verb_prompt -1 \
        --task_prompt 4 \
        --task_num 4 \
        --cgl_split_source full \
        --cgl_split_mode ratio \
        --cgl_head_ratio 0.08 \
        --cgl_tail_ratio 0.008
done

# --clip_text \
# --clip_image \
# --cb
# --focal
# --eqlv2
# --reverse \
#  --traditional_task_branches
#  --st_adapter\
#  --st_adapter_dim 512 \
#  --st_adapter_kernel_size 3 
# --vpt_prompt \
# --vpt_prompt_len 4 \
# --vpt_prompt_layers 4
# # 1. >8k / <0.8k
# --cgl_split_source full \
# --cgl_split_mode absolute \
# --cgl_head_threshold 8000 \
# --cgl_tail_threshold 800

# # 2. >10k / <1k
# --cgl_split_source full \
# --cgl_split_mode absolute \
# --cgl_head_threshold 10000 \
# --cgl_tail_threshold 1000

# # 3. >12k / <1.2k
# --cgl_split_source full \
# --cgl_split_mode absolute \
# --cgl_head_threshold 12000 \
# --cgl_tail_threshold 1200

# # 4. Top/Bottom 10%
# --cgl_split_source full \
# --cgl_split_mode percentile \
# --cgl_head_percent 0.10 \
# --cgl_tail_percent 0.10