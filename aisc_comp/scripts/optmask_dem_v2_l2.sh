

attacks=(optmask_dem_v2_l2)
# steps_list=(60)
steps_list=(30 40 50 60 70 90 120 150)
alpha_list=(0.25)
# alpha_list=(0.22 0.25 0.28 0.3)
mtypes=(comp_keypoint)
ns=(5)
# bounds=(1.5 2 2.5 3)
bounds=(1.5 2 2.5 3)
# bounds=(1.5 2.5)
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
                        nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
                        --attack_name $attack \
                        --batch_size 5 \
                        --steps $step \
                        --alpha $alpha \
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
    done
done