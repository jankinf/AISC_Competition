

attacks=(optmask_ds_l2)
steps_list=(30 40 50 60 70 90 120 150 200)
# alpha_list=(0.3 0.35 0.4)
# alpha_list=(0.25 0.28 0.29 0.31 0.32)
alpha_list=(0.25)
# mtypes=(raw_keypoint comp_keypoint hiltert_m5 hiltert_m7)
mtypes=(comp_keypoint)
# steps_list=(60)
# alpha_list=(0.4)
# mtypes=(hiltert_m7)
for attack in ${attacks[@]}
do
    for step in ${steps_list[@]}
    do
        for alpha in ${alpha_list[@]}
        do
            for mtype in ${mtypes[@]}
            do
                nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
                --attack_name $attack \
                --batch_size 10 \
                --steps $step \
                --alpha $alpha \
                --masktype keypoints \
                --mtype $mtype \
                --input_dir /data/projects/aisc_facecomp/data \
                --output_dir ./results/${attack}@${mtype}_alpha${alpha}_steps${step}_ens4_model0178 \
                --model_idx 0 1 7 8 \
                --device 0 &
                sleep 5s
            done
        done
    done
done
