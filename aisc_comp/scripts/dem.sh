
attacks=(dem)
steps_list=(20)
eps_list=(75 100 125 150 175 200)
for attack in ${attacks[@]}
do
    for eps in ${eps_list[@]}
    do
        for step in ${steps_list[@]}
        do
            nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
            --attack_name $attack \
            --batch_size 20 \
            --max_epsilon $eps \
            --steps $step \
            --input_dir /data/projects/aisc_facecomp/data \
            --output_dir ./results/${attack}_eps${eps}_steps${step} \
            --device 0 &
        done
    done
done
attacks=(dem)
steps_list=(10 20 30 40 50 60)
eps_list=(50)
for attack in ${attacks[@]}
do
    for eps in ${eps_list[@]}
    do
        for step in ${steps_list[@]}
        do
            nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
            --attack_name $attack \
            --batch_size 20 \
            --max_epsilon $eps \
            --steps $step \
            --input_dir /data/projects/aisc_facecomp/data \
            --output_dir ./results/${attack}_eps${eps}_steps${step} \
            --device 0 &
        done
    done
done
wait
attacks=(dem)
steps_list=(10 20 30 40 50 60)
eps_list=(75)
for attack in ${attacks[@]}
do
    for eps in ${eps_list[@]}
    do
        for step in ${steps_list[@]}
        do
            nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
            --attack_name $attack \
            --batch_size 20 \
            --max_epsilon $eps \
            --steps $step \
            --input_dir /data/projects/aisc_facecomp/data \
            --output_dir ./results/${attack}_eps${eps}_steps${step} \
            --device 0 &
        done
    done
done







