ckpt='./ckpt/robust.pth.tar'
# ckpt='./ckpt/ckpt_iter.pth.tar'

# epslist=(1 2 3 4 5 6 7 8 10 12 14 16)
# cpu=2
# memory=16394
cpu=1
memory=2394
epslist=(3 4 5 6 8)
train_images=300
for eps in "${epslist[@]}"
do
    label=1
    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch 10 \
    --max_epsilon ${eps} \
    --batch_size 50 \
    --train_images ${train_images} \
    --loss_type celoss \
    --layername ReLU2596 \
    --beta 12 &

    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch 10 \
    --max_epsilon ${eps} \
    --batch_size 50 \
    --train_images ${train_images} \
    --loss_type layernorm \
    --layername ReLU2596 \
    --beta 12 & 

    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch 10 \
    --max_epsilon ${eps} \
    --batch_size 50 \
    --train_images ${train_images} \
    --loss_type layernorm \
    --layername ReLU3156 \
    --beta 12 & 

    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch 10 \
    --max_epsilon ${eps} \
    --batch_size 50 \
    --train_images ${train_images} \
    --loss_type layernorm \
    --layername ReLU2442 \
    --beta 12 & 

    label=0
    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch 10 \
    --max_epsilon ${eps} \
    --batch_size 50 \
    --train_images ${train_images} \
    --loss_type celoss \
    --layername ReLU2596 \
    --beta 12 &

    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch 10 \
    --max_epsilon ${eps} \
    --batch_size 50 \
    --train_images ${train_images} \
    --loss_type layernorm \
    --layername ReLU2596 \
    --beta 12 & 

    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch 10 \
    --max_epsilon ${eps} \
    --batch_size 50 \
    --train_images ${train_images} \
    --loss_type layernorm \
    --layername ReLU3156 \
    --beta 12 & 

    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch 10 \
    --max_epsilon ${eps} \
    --batch_size 50 \
    --train_images ${train_images} \
    --loss_type layernorm \
    --layername ReLU2442 \
    --beta 12 & 
done

wait
epslist=(3 4 5 6 8)
train_images=500
for eps in "${epslist[@]}"
do
    label=1
    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch 10 \
    --max_epsilon ${eps} \
    --batch_size 50 \
    --train_images ${train_images} \
    --loss_type celoss \
    --layername ReLU2596 \
    --beta 12 &

    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch 10 \
    --max_epsilon ${eps} \
    --batch_size 50 \
    --train_images ${train_images} \
    --loss_type layernorm \
    --layername ReLU2596 \
    --beta 12 & 

    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch 10 \
    --max_epsilon ${eps} \
    --batch_size 50 \
    --train_images ${train_images} \
    --loss_type layernorm \
    --layername ReLU3156 \
    --beta 12 & 

    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch 10 \
    --max_epsilon ${eps} \
    --batch_size 50 \
    --train_images ${train_images} \
    --loss_type layernorm \
    --layername ReLU2442 \
    --beta 12 & 

    label=0
    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch 10 \
    --max_epsilon ${eps} \
    --batch_size 50 \
    --train_images ${train_images} \
    --loss_type celoss \
    --layername ReLU2596 \
    --beta 12 &

    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch 10 \
    --max_epsilon ${eps} \
    --batch_size 50 \
    --train_images ${train_images} \
    --loss_type layernorm \
    --layername ReLU2596 \
    --beta 12 & 

    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch 10 \
    --max_epsilon ${eps} \
    --batch_size 50 \
    --train_images ${train_images} \
    --loss_type layernorm \
    --layername ReLU3156 \
    --beta 12 & 

    nohup rlaunch --cpu ${cpu} --memory ${memory} -- python -m attack.methods.whitebox.uap_sgd \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --label ${label} \
    --step_decay 0.6 \
    --epoch 10 \
    --max_epsilon ${eps} \
    --batch_size 50 \
    --train_images ${train_images} \
    --loss_type layernorm \
    --layername ReLU2442 \
    --beta 12 & 
done