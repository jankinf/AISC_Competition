

attacks=(optmask_ct_catfeat_fill)
# 5 10 20 30 40 60 100 150 200
steps_list=(100)
alpha_list=(1.6)
for attack in ${attacks[@]}
do
    for alpha in ${alpha_list[@]}
    do
        for step in ${steps_list[@]}
        do
            # for idx in 0 1 2 3 4 5 6 7
            for idx in 1 7
            do
                nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
                --attack_name $attack \
                --batch_size 10 \
                --alpha $alpha \
                --steps $step \
                --input_dir /data/projects/aisc_facecomp/data \
                --output_dir ./results/${attack}_alpha${alpha}_steps${step}_ens4_model0178 \
                --model_idx 0 1 7 8 \
                --sep 8 \
                --idx $idx \
                --device 0 &
            done
            # python run.py \
            # --attack_name $attack \
            # --batch_size 10 \
            # --alpha $alpha \
            # --steps $step \
            # --input_dir /data/projects/aisc_facecomp/data \
            # --output_dir ./results/${attack}_alpha${alpha}_steps${step}_ens4_model0178 \
            # --model_idx 0 1 7 8 \
            # --device 0
        done
    done
done
