

# for eps in 3 4 5 6 7 8
# for eps in 3 4 5 6 7 8
for eps in 8
do
    # for n in 1 2 3 4 5 6 7 8 9 10
    for bound in 3 4 5 6 7
    do
        for n in 20 30
        do
            nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.whitebox.de_mi_fgsm \
            --pretrain vggface2 \
            --num_iter 10 \
            --bound $bound \
            --n $n \
            --max_epsilon $eps &
        done
    done
    # nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.whitebox.de_mi_fgsm \
    # --pretrain casia-webface \
    # --num_iter 10 \
    # --max_epsilon $eps &
done