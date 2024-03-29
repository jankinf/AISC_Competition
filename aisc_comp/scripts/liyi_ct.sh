# attack=ct
# steps_list=(100)
# alphas=(0.3)
# decay=1.0
# resize_rate=(1.3)
# diversity_prob=1.0
# mtype=comp_v2
# n_ens=7
# for steps in ${steps_list[@]}
# do
#     for alpha in ${alphas[@]}
#     do
#         for rr in ${resize_rate[@]}
#         do
#             for idx in 0 1 2 3 4 5 6 7
#             do
#                 # nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attacks.liyi.run \
#                 CUDA_VISIBLE_DEVICES=$idx python -m attacks.liyi.run \
#                 --attack_name $attack --batch_size 2 --steps $steps \
#                 --decay $decay --alpha $alpha \
#                 --resize_rate $rr --diversity_prob $diversity_prob \
#                 --input_dir /data/projects/aisc_facecomp/data \
#                 --n_ens $n_ens \
#                 --speedup \
#                 --sep 8 \
#                 --idx $idx \
#                 --model_idx 0 1 3 7 8 \
#                 --mtype $mtype \
#                 --output_dir ./results/liyi/${attack}_steps${steps}_alpha${alpha}_rr${rr}_dp${diversity_prob}_d${n_ens}_${mtype}_ord2_ens01378 \
#                 --use_di >> logs/${attack}_steps${steps}_alpha${alpha}_rr${rr}_dp${diversity_prob}_d${n_ens}_${mtype}_ord2_ens01378.log &
#             done
#         done
#     done
# done

# 2 gpu: 测试10 11 12 13 四个模型
# attack=ct
# steps_list=(100)
# alphas=(0.3)
# decay=1.0
# resize_rate=(1.3)
# diversity_prob=1.0
# mtype=comp_v2
# n_ens=4
# model_idx=11
# for steps in ${steps_list[@]}
# do
#     for alpha in ${alphas[@]}
#     do
#         for rr in ${resize_rate[@]}
#         do
#             for idx in 0 1
#             do
#                 # nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attacks.liyi.run \
#                 CUDA_VISIBLE_DEVICES=$idx python -m attacks.liyi.run \
#                 --attack_name $attack --batch_size 4 --steps $steps \
#                 --decay $decay --alpha $alpha \
#                 --resize_rate $rr --diversity_prob $diversity_prob \
#                 --input_dir /data/projects/aisc_facecomp/data \
#                 --n_ens $n_ens \
#                 --speedup \
#                 --sep 2 \
#                 --idx $idx \
#                 --model_idx $model_idx \
#                 --mtype $mtype \
#                 --output_dir ./results/liyi/${attack}_steps${steps}_alpha${alpha}_rr${rr}_dp${diversity_prob}_d${n_ens}_${mtype}_ord2_model${model_idx} \
#                 --use_di >> logs/${attack}_steps${steps}_alpha${alpha}_rr${rr}_dp${diversity_prob}_d${n_ens}_${mtype}_ord2_model${model_idx}.log &
#                 # --output_dir ./results/liyi/${attack}_steps${steps}_alpha${alpha}_rr${rr}_dp${diversity_prob}_d${n_ens}_${mtype}_ord2_model_10_11_12_13 \
#                 # --use_di >> logs/${attack}_steps${steps}_alpha${alpha}_rr${rr}_dp${diversity_prob}_d${n_ens}_${mtype}_ord2_model_10_11_12_13.log &
#                 # --model_idx 10 11 12 13 \
#             done
#         done
#     done
# done

attack=ct
# steps_list=(110 120 130)
steps_list=(120)
alphas=(0.25)
decay=1.0
resize_rate=(1.3)
diversity_prob=1.0
# mtype=comp_v2
# mtype=origin_v2
# mtype=comp_tmode_v2
# mtype=origin_tmode
# mtype=origin_tmode_v2
# mtype=comp_tmode
# mtype=comp_tmode_v3
mtype=fixed_mask_v2
n_ens=7
for steps in ${steps_list[@]}
do
    for alpha in ${alphas[@]}
    do
        for rr in ${resize_rate[@]}
        do
            for idx in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
            do
                CUDA_VISIBLE_DEVICES=$idx python -m attacks.liyi.run \
                --attack_name $attack --batch_size 2 --steps $steps \
                --decay $decay --alpha $alpha \
                --resize_rate $rr --diversity_prob $diversity_prob \
                --input_dir /data/projects/aisc_facecomp/data \
                --n_ens $n_ens \
                --speedup \
                --sep 16 \
                --idx $idx \
                --model_idx 0 1 3 4 7 8 13 15 \
                --ori_mask \
                --mtype $mtype \
                --output_dir ./results/liyi/${attack}_steps${steps}_alpha${alpha}_rr${rr}_dp${diversity_prob}_d${n_ens}_${mtype}_ori_mask_ord2_model0134781315 \
                --use_di > logs/nohup.log &
                # --output_dir ./results/liyi/${attack}_steps${steps}_alpha${alpha}_rr${rr}_dp${diversity_prob}_d${n_ens}_${mtype}_ord2_model0134781315 \
                # --use_di >> logs/${attack}_steps${steps}_alpha${alpha}_rr${rr}_dp${diversity_prob}_d${n_ens}_${mtype}_ord2_model01347813.log &
            done
        done
    done
done