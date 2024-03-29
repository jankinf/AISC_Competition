attacks=(optmask_dem_v2)
steps_list=(5)
eps_list=(40)
for attack in ${attacks[@]}
do
    for eps in ${eps_list[@]}
    do
        for step in ${steps_list[@]}
        do
            for bound in 7
            do
                for n in 20
                do
                    for idx in 0 1 2 3
                    do
                        nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
                        --attack_name $attack \
                        --batch_size 20 \
                        --max_epsilon $eps \
                        --steps $step \
                        --n $n \
                        --bound $bound \
                        --input_dir /data/projects/aisc_facecomp/data \
                        --output_dir ./results/${attack}_eps${eps}_steps${step}_bound${bound}_n${n}_ens1 \
                        --sep 4 \
                        --idx $idx \
                        --model_idx 0 \
                        --device 0 &
                    done
                done
            done
        done
    done
done
