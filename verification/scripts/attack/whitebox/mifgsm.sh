ckpt='./ckpt/robust.pth.tar'
# ckpt='./ckpt/ckpt_iter.pth.tar'

epslist=(1 2 3 4 5 6 7 8 10 12 14 16)
for eps in "${epslist[@]}"
do
    nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.whitebox.mi_fgsm \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --max_epsilon ${eps} \
    --num_iter 10 \
    --label 1 &

    nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.whitebox.mi_fgsm \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --max_epsilon ${eps} \
    --num_iter 10 \
    --label 0 &
done