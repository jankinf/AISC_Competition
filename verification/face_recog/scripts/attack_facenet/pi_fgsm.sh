

# for ampf in 1 1.5 2 2.5 3 5 8 10
for ampf in 1.5 2 2.5 3 5 10
do
    for eps in 7 8
    do
        nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.whitebox.pi_fgsm \
        --ampf $ampf \
        --pretrain vggface2 \
        --num_iter 10 \
        --max_epsilon $eps &

        # nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.whitebox.pi_fgsm \
        # --ampf $ampf \
        # --pretrain casia-webface \
        # --num_iter 10 \
        # --max_epsilon $eps &
    done
done