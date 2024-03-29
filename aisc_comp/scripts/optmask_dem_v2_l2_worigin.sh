

attacks=(optmask_dem_v2_l2)
steps_list=(20 25 30 60)
# steps_list=(20 25 30)
alpha_list=(0.25)
mtypes=(comp_keypoint)
# ns=(9) # bs 3
ns=(4) # bs 5
# ns=(5 6 7) # bs 5
bounds=(1.5)
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
                    for bound in ${bounds[@]}
                    do
                        # nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
                        # --attack_name $attack \
                        # --batch_size 5 \
                        # --steps $step \
                        # --alpha $alpha \
                        # --masktype keypoints \
                        # --mtype $mtype \
                        # --n $n \
                        # --bound $bound \
                        # --input_dir /data/projects/aisc_facecomp/data \
                        # --output_dir ./results/${attack}_worigin@${mtype}_alpha${alpha}_steps${step}_n${n}_bound${bound}ens4_model0178 \
                        # --model_idx 0 1 7 8 \
                        # --device 0 &
                        # sleep 5s
                        nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
                        --attack_name $attack \
                        --batch_size 1 \
                        --steps $step \
                        --alpha $alpha \
                        --masktype keypoints \
                        --mtype $mtype \
                        --n $n \
                        --bound $bound \
                        --input_dir /data/projects/aisc_facecomp/data \
                        --output_dir ./results/${attack}_worigin@${mtype}_alpha${alpha}_steps${step}_n${n}_bound${bound}ens7_model0126789 \
                        --model_idx 0 1 2 6 7 8 9 \
                        --device 0 &
                        sleep 5s
                    done
                done
            done
        done
    done
done