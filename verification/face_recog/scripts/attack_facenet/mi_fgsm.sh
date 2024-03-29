

# for eps in 3 4 5 6 7 8
for eps in 7 8
do
    nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.whitebox.mi_fgsm \
    --pretrain vggface2 \
    --num_iter 10 \
    --max_epsilon $eps &

    # nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.whitebox.mi_fgsm \
    # --pretrain casia-webface \
    # --num_iter 10 \
    # --max_epsilon $eps &
done
