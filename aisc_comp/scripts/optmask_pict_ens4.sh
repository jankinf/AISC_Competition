
attacks=(optmask_pict)
steps_list=(20)
eps_list=(40)
for attack in ${attacks[@]}
do
    for eps in ${eps_list[@]}
    do
        for step in ${steps_list[@]}
        do
            for idx in 0 1
            do
                nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
                --attack_name $attack \
                --batch_size 1 \
                --max_epsilon $eps \
                --steps $step \
                --input_dir /data/projects/aisc_facecomp/data \
                --output_dir ./results/${attack}_eps${eps}_steps${step}_ens4_model${model_idx} \
                --speedup \
                --sep 2 \
                --idx $idx \
                --model_idx 0 1 7 8 \
                --device 0 &
            done
        done
    done
done
