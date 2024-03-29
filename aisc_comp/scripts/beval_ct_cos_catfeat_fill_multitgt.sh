

attacks=(beval_ct_cos_catfeat_fill_multitgt)
# 5 10 20 30 40 60 100 150 200
# steps_list=(50 100 200)
# alpha_list=(0.8 1.6 3.2)
# for attack in ${attacks[@]}
# do
#     for alpha in ${alpha_list[@]}
#     do
#         for step in ${steps_list[@]}
#         do
#             nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
#             --attack_name $attack \
#             --batch_size 1 \
#             --alpha $alpha \
#             --steps $step \
#             --input_dir /data/projects/aisc_facecomp/data \
#             --output_dir ./results/${attack}_alpha${alpha}_steps${step}_ens4_model0178_bmodel2 \
#             --model_idx 0 1 7 8 \
#             --model_b_idx 2 \
#             --device 0 > logs/${attack}_alpha${alpha}_steps${step}_ens4_model0178_bmodel2.log &
#         done
#     done
# done
# steps_list=(50)
# alpha_list=(1.2 1.6 3.2)
# for attack in ${attacks[@]}
# do
#     for alpha in ${alpha_list[@]}
#     do
#         for step in ${steps_list[@]}
#         do
#             # nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
#             # --attack_name $attack \
#             # --batch_size 1 \
#             # --alpha $alpha \
#             # --steps $step \
#             # --input_dir /data/projects/aisc_facecomp/data \
#             # --output_dir ./results/${attack}_alpha${alpha}_steps${step}_ens9_model013456789_bmodel2 \
#             # --model_idx 0 1 3 4 5 6 7 8 9 \
#             # --model_b_idx 2 \
#             # --device 0 > logs/${attack}_alpha${alpha}_steps${step}_ens9_model013456789_bmodel2.log &
#             # sleep 5s
#             # nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
#             # --attack_name $attack \
#             # --batch_size 1 \
#             # --alpha $alpha \
#             # --steps $step \
#             # --input_dir /data/projects/aisc_facecomp/data \
#             # --output_dir ./results/${attack}_alpha${alpha}_steps${step}_ens4_model0178_bmodel234569 \
#             # --model_idx 0 1 7 8 \
#             # --model_b_idx 2 3 4 5 6 9 \
#             # --device 0 > logs/${attack}_alpha${alpha}_steps${step}_ens4_model0178_bmodel234569.log &
#             # sleep 5s
#             nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
#             --attack_name $attack \
#             --batch_size 1 \
#             --alpha $alpha \
#             --steps $step \
#             --input_dir /data/projects/aisc_facecomp/data \
#             --output_dir ./results/${attack}_alpha${alpha}_steps${step}_ens4_model0178_bmodel347 \
#             --model_idx 0 1 7 8 \
#             --model_b_idx 3 4 7 \
#             --device 0 > logs/${attack}_alpha${alpha}_steps${step}_ens4_model0178_bmodel347.log &
#             sleep 5s
#         done
#     done
# done


steps_list=(50)
alpha_list=(1.2 3.2)
for attack in ${attacks[@]}
do
    for alpha in ${alpha_list[@]}
    do
        for step in ${steps_list[@]}
        do
            nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
            --attack_name $attack \
            --batch_size 1 \
            --alpha $alpha \
            --steps $step \
            --input_dir /data/projects/aisc_facecomp/data \
            --output_dir ./results/${attack}_alpha${alpha}_steps${step}_ens4_model0178_bmodel347 \
            --model_idx 0 1 7 8 \
            --model_b_idx 3 4 7 \
            --device 0 > logs/${attack}_alpha${alpha}_steps${step}_ens4_model0178_bmodel347.log &
            sleep 5s
        done
    done
done


