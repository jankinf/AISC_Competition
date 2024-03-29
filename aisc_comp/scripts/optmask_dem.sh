# ============================================ old version ============================================
# attacks=(optmask_dem)
# steps_list=(20 30 40 50 75 100)
# eps_list=(40)
# for attack in ${attacks[@]}
# do
#     for eps in ${eps_list[@]}
#     do
#         for step in ${steps_list[@]}
#         do
#             nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
#             --attack_name $attack \
#             --batch_size 20 \
#             --max_epsilon $eps \
#             --steps $step \
#             --input_dir /data/projects/aisc_facecomp/data \
#             --output_dir ./results/${attack}_eps${eps}_steps${step} \
#             --device 0 &
#         done
#     done
# done
# attacks=(optmask_dem)
# steps_list=(20 30 40 50 75 100)
# eps_list=(45)
# for attack in ${attacks[@]}
# do
#     for eps in ${eps_list[@]}
#     do
#         for step in ${steps_list[@]}
#         do
#             nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
#             --attack_name $attack \
#             --batch_size 20 \
#             --max_epsilon $eps \
#             --steps $step \
#             --input_dir /data/projects/aisc_facecomp/data \
#             --output_dir ./results/${attack}_eps${eps}_steps${step} \
#             --device 0 &
#         done
#     done
# done
# attacks=(optmask_dem)
# steps_list=(20 30 40 50 75 100)
# eps_list=(35)
# for attack in ${attacks[@]}
# do
#     for eps in ${eps_list[@]}
#     do
#         for step in ${steps_list[@]}
#         do
#             nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
#             --attack_name $attack \
#             --batch_size 20 \
#             --max_epsilon $eps \
#             --steps $step \
#             --input_dir /data/projects/aisc_facecomp/data \
#             --output_dir ./results/${attack}_eps${eps}_steps${step} \
#             --device 0 &
#         done
#     done
# done

# ============================================ old version ============================================



# ============================================ new version ============================================

attacks=(optmask_dem)
alpha_list=(0.00627)
steps_list=(40 50 60)
mtype=comp_keypoint
ns=(4 5)
bound=1.5
for attack in ${attacks[@]}
do
    for alpha in ${alpha_list[@]}
    do
        for step in ${steps_list[@]}
        do
            for n in ${ns[@]}
            do
                nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
                --attack_name $attack \
                --batch_size 5 \
                --alpha $alpha \
                --steps $step \
                --masktype keypoints \
                --mtype $mtype \
                --n $n \
                --bound $bound \
                --use_mi \
                --input_dir /data/projects/aisc_facecomp/data \
                --output_dir ./results/${attack}_mi@${mtype}_alpha${alpha}_steps${step}_n${n}_bound${bound}ens4_model0178 \
                --model_idx 0 1 7 8 \
                --device 0 &
                sleep 5s
                
                nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
                --attack_name $attack \
                --batch_size 5 \
                --alpha $alpha \
                --steps $step \
                --masktype keypoints \
                --mtype $mtype \
                --n $n \
                --bound $bound \
                --input_dir /data/projects/aisc_facecomp/data \
                --output_dir ./results/${attack}@${mtype}_alpha${alpha}_steps${step}_n${n}_bound${bound}ens4_model0178 \
                --model_idx 0 1 7 8 \
                --device 0 &
                sleep 5s
            done
        done
    done
done


# ============================================ new version ============================================