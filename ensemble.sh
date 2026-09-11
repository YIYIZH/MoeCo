#!/bin/bash

for kfold in {4..4}
do
    version1_replaced=$(echo "241014_6_baseline_learnT_swinB_div4_p1_con1_k1_seed20000912" | sed "s/k1/k$kfold/g")
    echo "$version1_replaced"
    export CUDA_VISIBLE_DEVICES=1
    nohup python ensemble.py \
        --coe 0.4 \
        -t -e \
        --num_layers_PG 11 \
        --num_layers_R 4 \
        --num_R 3 \
        --seed 20000912 \
        --input_dim 1024 \
        --mask \
        --pos_w "pre_w" \
        --loss_type "none" \
        --decay_rate 0.99 \
        --dataset_variant cholect45-crossval \
        --kfold $kfold \
        --epochs 1000 \
        --batch 1 \
        -l 1e-2 5e-3 5e-3 \
        -w 9 18 400 \
        --version "t45ens-coe04f"$kfold \
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
        --target_prompt -1 \
        --verb_prompt -1 \
        --task_prompt 4 \
        --task_num 4 &
done

# --clip_text \
# --clip_image \
# --cb
# --focal
# --eqlv2
# --reverse \