
mode='attack'
label=1
nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.uniscore_momentum \
--i_net panorama.i_epoch_202622.th_0.8308.neupeak \
--train_images 500 \
--max_epsilon 4 \
--label ${label} &
# python -m attack.methods.uniscore_momentum \
# --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
# --train_images 500 \
# --max_epsilon 4 \
# --label ${label}

