

# for eps in 3 4 5 6 7 8
for eps in 8
do
    # for n in 0 1 2 3 4 5 6 7 8 9 10
    for n in 15
    do
        python -m attack.methods.whitebox.random_crop \
        --pretrain vggface2 \
        --num_iter 10 \
        --n $n \
        --max_epsilon $eps
        # nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.whitebox.random_crop \
        # --pretrain vggface2 \
        # --num_iter 10 \
        # --n $n \
        # --max_epsilon $eps &
    done

    # nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.whitebox.random_crop \
    # --pretrain casia-webface \
    # --num_iter 10 \
    # --max_epsilon $eps &
done