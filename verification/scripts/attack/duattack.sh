
mode='attack'
label=1
# nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.duattack \
# --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
# --train_images 500 \
# --max_epsilon 8 \
# --n_queries 10000 \
# --label ${label} &
# nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.duattack \
# --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
# --train_images 500 \
# --max_epsilon 4 \
# --n_queries 10000 \
# --label ${label} &
# nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.duattack \
# --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
# --train_images 500 \
# --max_epsilon 16 \
# --n_queries 10000 \
# --label ${label} &
nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.duattack \
--i_net panorama.i_epoch_202622.th_0.8308.neupeak \
--train_images 500 \
--max_epsilon 20 \
--n_queries 10000 \
--label ${label} &
