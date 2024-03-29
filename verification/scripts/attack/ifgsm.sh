ckpt='./ckpt/robust.pth.tar'
# ckpt='./ckpt/ckpt_iter.pth.tar'

epslist=(16 12 8 4 2 1)
for eps in "${epslist[@]}"
do
    nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.i_fgsm \
    --max_epsilon ${eps} --num_iter 10 \
    --workers 0 --batch_size 1 --target -1 \
    --ckpt ${ckpt} \
    --eval_ckpts \
    './ckpt/ckpt_iter.pth.tar' \
    './ckpt/robust.pth.tar' &

    nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.i_fgsm \
    --max_epsilon ${eps} --num_iter 10 \
    --workers 0 --batch_size 1 --target 1 \
    --ckpt ${ckpt} \
    --eval_ckpts \
    './ckpt/ckpt_iter.pth.tar' \
    './ckpt/robust.pth.tar' &

    nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.i_fgsm \
    --max_epsilon ${eps} --num_iter 10 \
    --workers 0 --batch_size 1 --target 0 \
    --ckpt ${ckpt} \
    --eval_ckpts \
    './ckpt/ckpt_iter.pth.tar' \
    './ckpt/robust.pth.tar' &
done