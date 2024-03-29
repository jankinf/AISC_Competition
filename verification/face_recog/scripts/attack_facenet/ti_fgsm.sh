

# for eps in 3 4 5 6 7 8
# do
#     nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.whitebox.ti_fgsm \
#     --pretrain vggface2 \
#     --num_iter 10 \
#     --max_epsilon $eps &

#     # nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.whitebox.ti_fgsm \
#     # --pretrain casia-webface \
#     # --num_iter 10 \
#     # --max_epsilon $eps &
# done

for kernel in 7 9 14 20
do
    nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.whitebox.ti_fgsm \
    --pretrain vggface2 \
    --kernel_size $kernel \
    --num_iter 10 \
    --max_epsilon 7 &

    nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.whitebox.ti_fgsm \
    --pretrain vggface2 \
    --kernel_size $kernel \
    --num_iter 10 \
    --max_epsilon 8 &

    # nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.whitebox.ti_fgsm \
    # --pretrain casia-webface \
    # --num_iter 10 \
    # --max_epsilon $eps &
done