ckpt='./ckpt/robust.pth.tar'
# ckpt='./ckpt/ckpt_iter.pth.tar'

# epslist=(1 2 3 4 5 6 7 8 10 12 14 16)
cpu=2
memory=16394
epslist=(3 4 5 6 8 10)
epoch=50
batch_size=50
for eps in "${epslist[@]}"
do
    train_images=300
    label=1
    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch ${epoch} \
    --max_epsilon ${eps} \
    --batch_size ${batch_size} \
    --train_images ${train_images} \
    --loss_type celoss \
    --layername ReLU2596 \
    --beta 12 &

    label=0
    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch ${epoch} \
    --max_epsilon ${eps} \
    --batch_size ${batch_size} \
    --train_images ${train_images} \
    --loss_type celoss \
    --layername ReLU2596 \
    --beta 12 &

    train_images=500
    label=1
    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch ${epoch} \
    --max_epsilon ${eps} \
    --batch_size ${batch_size} \
    --train_images ${train_images} \
    --loss_type celoss \
    --layername ReLU2596 \
    --beta 12 &


    label=0
    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch ${epoch} \
    --max_epsilon ${eps} \
    --batch_size ${batch_size} \
    --train_images ${train_images} \
    --loss_type celoss \
    --layername ReLU2596 \
    --beta 12 &

    train_images=1000
    label=1
    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch ${epoch} \
    --max_epsilon ${eps} \
    --batch_size ${batch_size} \
    --train_images ${train_images} \
    --loss_type celoss \
    --layername ReLU2596 \
    --beta 12 &


    label=0
    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch ${epoch} \
    --max_epsilon ${eps} \
    --batch_size ${batch_size} \
    --train_images ${train_images} \
    --loss_type celoss \
    --layername ReLU2596 \
    --beta 12 &
done

wait


batch_size=100
for eps in "${epslist[@]}"
do
    train_images=300
    label=1
    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch ${epoch} \
    --max_epsilon ${eps} \
    --batch_size ${batch_size} \
    --train_images ${train_images} \
    --loss_type celoss \
    --layername ReLU2596 \
    --beta 12 &

    label=0
    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch ${epoch} \
    --max_epsilon ${eps} \
    --batch_size ${batch_size} \
    --train_images ${train_images} \
    --loss_type celoss \
    --layername ReLU2596 \
    --beta 12 &

    train_images=500
    label=1
    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch ${epoch} \
    --max_epsilon ${eps} \
    --batch_size ${batch_size} \
    --train_images ${train_images} \
    --loss_type celoss \
    --layername ReLU2596 \
    --beta 12 &


    label=0
    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch ${epoch} \
    --max_epsilon ${eps} \
    --batch_size ${batch_size} \
    --train_images ${train_images} \
    --loss_type celoss \
    --layername ReLU2596 \
    --beta 12 &

    train_images=1000
    label=1
    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch ${epoch} \
    --max_epsilon ${eps} \
    --batch_size ${batch_size} \
    --train_images ${train_images} \
    --loss_type celoss \
    --layername ReLU2596 \
    --beta 12 &


    label=0
    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch ${epoch} \
    --max_epsilon ${eps} \
    --batch_size ${batch_size} \
    --train_images ${train_images} \
    --loss_type celoss \
    --layername ReLU2596 \
    --beta 12 &
done