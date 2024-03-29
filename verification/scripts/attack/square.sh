# 无目标攻击
mode='attack'
label=1
nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.square \
--i_net panorama.i_epoch_202622.th_0.8308.neupeak \
--image /data/datasets/fmp/imgs/${mode}/2019-07-22_2019-07-28_${mode}.nori.list \
--max_images 1000 --max_epsilon 4 \
--mode ${mode} \
--label ${label} &

mode='normal'
label=0
nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.square \
--i_net panorama.i_epoch_202622.th_0.8308.neupeak \
--image /data/datasets/fmp/imgs/${mode}/2019-07-22_2019-07-28_${mode}.nori.list \
--max_images 1000 --max_epsilon 4 \
--mode ${mode} \
--label ${label} &

# 有目标攻击 attack
mode='normal'
label=1
nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.square \
--i_net panorama.i_epoch_202622.th_0.8308.neupeak \
--image /data/datasets/fmp/imgs/${mode}/2019-07-22_2019-07-28_${mode}.nori.list \
--max_images 1000 --max_epsilon 4 \
--targeted \
--mode ${mode} \
--label ${label} &

# mode='attack'
# label=1
# nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.square \
# --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
# --image /data/datasets/fmp/imgs/${mode}/2019-07-22_2019-07-28_${mode}.nori.list \
# --max_images 1000 --max_epsilon 4 \
# --targeted \
# --mode ${mode} \
# --label ${label} > attack1.out &


# 有目标攻击 normal
# mode='normal'
# label=0
# nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.square \
# --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
# --image /data/datasets/fmp/imgs/${mode}/2019-07-22_2019-07-28_${mode}.nori.list \
# --max_images 1000 --max_epsilon 4 \
# --targeted \
# --mode ${mode} \
# --label ${label} > normal0.out &

mode='attack'
label=0
nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.square \
--i_net panorama.i_epoch_202622.th_0.8308.neupeak \
--image /data/datasets/fmp/imgs/${mode}/2019-07-22_2019-07-28_${mode}.nori.list \
--max_images 1000 --max_epsilon 4 \
--targeted \
--mode ${mode} \
--label ${label} &