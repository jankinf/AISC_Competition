

attacks=(optmask_bim_l2)
steps_list=(40 50 60 80 90)
mtypes=(comp_keypoint)
alpha_list=(0.25 0.28 0.3)
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
                --device 0 > logs/${attack}@${mtype}_alpha${alpha}_steps${step}_ens4_model0178.log &
                sleep 5s

                nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
                --attack_name $attack \
                --batch_size 1 \
                --steps $step \
                --alpha $alpha \
                --masktype keypoints \
                --mtype $mtype \
                --input_dir /data/projects/aisc_facecomp/data \
                --output_dir ./results/${attack}@${mtype}_alpha${alpha}_steps${step}_ens7_model0126789 \
                --model_idx 0 1 2 6 7 8 9 \
                --device 0 > logs/${attack}@${mtype}_alpha${alpha}_steps${step}_ens7_model0126789.log &
                sleep 5s
            done
        done
    done
done
