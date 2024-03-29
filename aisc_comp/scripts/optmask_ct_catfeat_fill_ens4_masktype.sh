

attacks=(optmask_ct_catfeat_fill)
# masktypes=(square_center square_top )
masktypes=(square_bottom)
steps_list=(50)
alpha_list=(1.6)
for attack in ${attacks[@]}
do
    for alpha in ${alpha_list[@]}
    do
        for step in ${steps_list[@]}
        do
            for masktype in ${masktypes[@]}
            do
                for idx in 0
                do
                    nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
                    --attack_name $attack \
                    --batch_size 10 \
                    --alpha $alpha \
                    --steps $step \
                    --masktype ${masktype} \
                    --input_dir /data/projects/aisc_facecomp/data \
                    --output_dir ./results/${attack}@${masktype}_alpha${alpha}_steps${step}_ens4_model0178 \
                    --model_idx 0 1 7 8 \
                    --sep 2 \
                    --idx $idx \
                    --device 0 > logs/${attack}@${masktype}_alpha${alpha}_steps${step}_ens4_model0178.log&
                    sleep 5s
                done
            done
        done
    done
done
