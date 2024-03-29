

attacks=(optmask_pict_catfeat_fill)
steps_list=(50)
eps_list=(40)
for attack in ${attacks[@]}
do
    for eps in ${eps_list[@]}
    do
        for step in ${steps_list[@]}
        do
            # for model_idx in 0 1 7 8
            for model_idx in 2
            do
                # nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
                # --attack_name $attack \
                # --batch_size 10 \
                # --max_epsilon $eps \
                # --steps $step \
                # --input_dir /data/projects/aisc_facecomp/data \
                # --output_dir ./results/${attack}_eps${eps}_steps${step}_ens1_model${model_idx} \
                # --model_idx $model_idx \
                # --device 0 &
                # sleep 5s
                python run.py \
                --attack_name $attack \
                --batch_size 10 \
                --max_epsilon $eps \
                --steps $step \
                --input_dir /data/projects/aisc_facecomp/data \
                --output_dir ./results/${attack}_eps${eps}_steps${step}_ens1_model${model_idx} \
                --model_idx $model_idx \
                --device 0
            done
        done
    done
done
