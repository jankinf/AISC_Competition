
# attacks=(optmask_pict)
# steps_list=(20)
# eps_list=(40)
# for attack in ${attacks[@]}
# do
#     for eps in ${eps_list[@]}
#     do
#         for step in ${steps_list[@]}
#         do
#             for idx in 0 1 2 3 4 5 6 7 8 9 10 11
#             do
#                 nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
#                 --attack_name $attack \
#                 --batch_size 1 \
#                 --max_epsilon $eps \
#                 --steps $step \
#                 --input_dir /data/projects/aisc_facecomp/data \
#                 --output_dir ./results/${attack}_eps${eps}_steps${step}_ens10 \
#                 --speedup \
#                 --sep 12 \
#                 --idx $idx \
#                 --device 0 &
#             done
#         done
#     done
# done

attacks=(optmask_pict)
steps_list=(20)
eps_list=(40)
for attack in ${attacks[@]}
do
    for eps in ${eps_list[@]}
    do
        for step in ${steps_list[@]}
        do
            for idx in 8 9 10 11
            do
                nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
                --attack_name $attack \
                --batch_size 1 \
                --max_epsilon $eps \
                --steps $step \
                --input_dir /data/projects/aisc_facecomp/data \
                --output_dir ./results/${attack}_eps${eps}_steps${step}_ens10 \
                --speedup \
                --sep 50 \
                --idx $idx \
                --device 0 &
            done
        done
    done
done

# attacks=(optmask_pict)
# steps_list=(20 30 40 50)
# eps_list=(40 45 35)
# for attack in ${attacks[@]}
# do
#     for eps in ${eps_list[@]}
#     do
#         for step in ${steps_list[@]}
#         do
#             nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python run.py \
#             --attack_name $attack \
#             --batch_size 1 \
#             --max_epsilon $eps \
#             --steps $step \
#             --input_dir /data/projects/aisc_facecomp/data \
#             --output_dir ./results/${attack}_eps${eps}_steps${step}_ens10 \
#             --device 0 &
#         done
#     done
# done