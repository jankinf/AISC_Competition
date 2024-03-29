
attacks=(optmask_clip_pict)
steps_list=(20 30 40 50)
eps_list=(40)
for attack in ${attacks[@]}
do
    for eps in ${eps_list[@]}
    do
        for step in ${steps_list[@]}
        do
            nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
            --attack_name $attack \
            --batch_size 10 \
            --max_epsilon $eps \
            --steps $step \
            --input_dir /data/projects/aisc_facecomp/data \
            --output_dir ./results/${attack}_eps${eps}_steps${step} \
            --device 0 &
        done
    done
done
attacks=(optmask_clip_pict)
steps_list=(20 30 40 50)
eps_list=(45)
for attack in ${attacks[@]}
do
    for eps in ${eps_list[@]}
    do
        for step in ${steps_list[@]}
        do
            nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
            --attack_name $attack \
            --batch_size 10 \
            --max_epsilon $eps \
            --steps $step \
            --input_dir /data/projects/aisc_facecomp/data \
            --output_dir ./results/${attack}_eps${eps}_steps${step} \
            --device 0 &
        done
    done
done
attacks=(optmask_clip_pict)
steps_list=(20 30 40 50)
eps_list=(35)
for attack in ${attacks[@]}
do
    for eps in ${eps_list[@]}
    do
        for step in ${steps_list[@]}
        do
            nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
            --attack_name $attack \
            --batch_size 10 \
            --max_epsilon $eps \
            --steps $step \
            --input_dir /data/projects/aisc_facecomp/data \
            --output_dir ./results/${attack}_eps${eps}_steps${step} \
            --device 0 &
        done
    done
done