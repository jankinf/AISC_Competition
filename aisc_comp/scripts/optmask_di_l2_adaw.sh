

attacks=(optmask_di_l2_adaw)
steps_list=(60)
# mtypes=(comp_keypoint)
# mtypes=(comp_keypoint_v2 comp_keypoint_2p_2eye comp_keypoint_3p_2eye_1mouse comp_keypoint_3p_2eye_1nose)
mtypes=(comp_keypoint_v2)
# mtypes=(raw_keypoint)
alpha_list=(0.3)
ns=(4)
resize_rates=(1.15)
# diversity_probs=(0.7 0.8)
diversity_probs=(0.7)
tems=(1 2 3 4)
for attack in ${attacks[@]}
do
    for step in ${steps_list[@]}
    do
        for alpha in ${alpha_list[@]}
        do
            for mtype in ${mtypes[@]}
            do
                for n in ${ns[@]}
                do
                    for resize_rate in ${resize_rates[@]}
                    do
                        for diversity_prob in ${diversity_probs[@]}
                        do
                            for tem in ${tems[@]}
                            do
                                nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
                                --attack_name $attack \
                                --batch_size 5 \
                                --steps $step \
                                --alpha $alpha \
                                --masktype keypoints \
                                --mtype $mtype \
                                --n $n \
                                --resize_rate $resize_rate \
                                --diversity_prob $diversity_prob \
                                --temperature $tem \
                                --input_dir /data/projects/aisc_facecomp/data \
                                --output_dir ./results/${attack}@${mtype}_alpha${alpha}_steps${step}_n${n}_rr${resize_rate}_dp${diversity_prob}_tem${tem}_ens4_model0178 \
                                --model_idx 0 1 7 8 \
                                --device 0  > logs/${attack}@${mtype}_alpha${alpha}_steps${step}_n${n}_rr${resize_rate}_dp${diversity_prob}_tem${tem}_ens4_model0178.log &
                                sleep 5s
                            done
                        done
                    done
                done
            done
        done
    done
done
