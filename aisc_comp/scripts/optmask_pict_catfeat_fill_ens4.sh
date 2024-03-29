

attacks=(optmask_pict_catfeat_fill)
steps_list=(50)
eps_list=(40)
for attack in ${attacks[@]}
do
    for eps in ${eps_list[@]}
    do
        for step in ${steps_list[@]}
        do
            # nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
            # --attack_name $attack \
            # --batch_size 10 \
            # --max_epsilon $eps \
            # --steps $step \
            # --input_dir /data/projects/aisc_facecomp/data \
            # --output_dir ./results/${attack}_eps${eps}_steps${step}_ens4_model0178 \
            # --model_idx 0 1 7 8 \
            # --device 0 &
            python run.py \
            --attack_name $attack \
            --batch_size 1 \
            --max_epsilon $eps \
            --steps $step \
            --input_dir /data/projects/aisc_facecomp/data \
            --output_dir ./results/debug/${attack}_eps${eps}_steps${step}_ens4_model0178 \
            --model_idx 0 1 7 8 \
            --device 0
        done
    done
done
