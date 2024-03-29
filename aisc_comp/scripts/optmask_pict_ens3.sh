
attacks=(optmask_pict)
steps_list=(20)
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
            --output_dir ./results/${attack}_eps${eps}_steps${step}_ens3_model017 \
            --model_idx 0 1 7 \
            --device 0 &

            nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
            --attack_name $attack \
            --batch_size 10 \
            --max_epsilon $eps \
            --steps $step \
            --input_dir /data/projects/aisc_facecomp/data \
            --output_dir ./results/${attack}_eps${eps}_steps${step}_ens3_model018 \
            --model_idx 0 1 8 \
            --device 0 &

            nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
            --attack_name $attack \
            --batch_size 10 \
            --max_epsilon $eps \
            --steps $step \
            --input_dir /data/projects/aisc_facecomp/data \
            --output_dir ./results/${attack}_eps${eps}_steps${step}_ens3_model078 \
            --model_idx 0 7 8 \
            --device 0 &

            nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
            --attack_name $attack \
            --batch_size 10 \
            --max_epsilon $eps \
            --steps $step \
            --input_dir /data/projects/aisc_facecomp/data \
            --output_dir ./results/${attack}_eps${eps}_steps${step}_ens3_model178 \
            --model_idx 1 7 8 \
            --device 0 &
            
            nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
            --attack_name $attack \
            --batch_size 10 \
            --max_epsilon $eps \
            --steps $step \
            --input_dir /data/projects/aisc_facecomp/data \
            --output_dir ./results/${attack}_eps${eps}_steps${step}_ens2_model78 \
            --model_idx 7 8 \
            --device 0 &

            nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
            --attack_name $attack \
            --batch_size 10 \
            --max_epsilon $eps \
            --steps $step \
            --input_dir /data/projects/aisc_facecomp/data \
            --output_dir ./results/${attack}_eps${eps}_steps${step}_ens2_model07 \
            --model_idx 0 7 \
            --device 0 &

            nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
            --attack_name $attack \
            --batch_size 10 \
            --max_epsilon $eps \
            --steps $step \
            --input_dir /data/projects/aisc_facecomp/data \
            --output_dir ./results/${attack}_eps${eps}_steps${step}_ens2_model08 \
            --model_idx 0 8 \
            --device 0 &

            nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
            --attack_name $attack \
            --batch_size 10 \
            --max_epsilon $eps \
            --steps $step \
            --input_dir /data/projects/aisc_facecomp/data \
            --output_dir ./results/${attack}_eps${eps}_steps${step}_ens2_model17 \
            --model_idx 1 7 \
            --device 0 &

            nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
            --attack_name $attack \
            --batch_size 10 \
            --max_epsilon $eps \
            --steps $step \
            --input_dir /data/projects/aisc_facecomp/data \
            --output_dir ./results/${attack}_eps${eps}_steps${step}_ens2_model18 \
            --model_idx 1 8 \
            --device 0 &
        done
    done
done
