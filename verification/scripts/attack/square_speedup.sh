
# start_ids=(0 200 400 600 800)
# start_ids=(0 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950)
# start_ids=(25 75 125 175 225 275 325 375 425 475 525 575 625 675 725 775 825 875 925 975)
# start_ids=(210 160 260 110)
# start_ids=(215 165 265 115)
# start_ids=(220 170 270 120)
# start_ids=(205 155 255 105)
eps=1
for start_id in "${start_ids[@]}"
do
    # 无目标攻击
    # mode='attack'
    # label=1
    # nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.square \
    # --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    # --image /data/datasets/fmp/imgs/${mode}/2019-07-22_2019-07-28_${mode}.nori.list \
    # --max_images 1000 --max_epsilon ${eps} \
    # --speedup \
    # --start ${start_id} \
    # --cnt 50 \
    # --mode ${mode} \
    # --label ${label} &

    # mode='normal'
    # label=0
    # nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.square \
    # --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    # --image /data/datasets/fmp/imgs/${mode}/2019-07-22_2019-07-28_${mode}.nori.list \
    # --max_images 1000 --max_epsilon ${eps} \
    # --speedup \
    # --start ${start_id} \
    # --cnt 50 \
    # --mode ${mode} \
    # --label ${label} &

    # 有目标攻击 attack
    mode='normal'
    label=1
    nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.square \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --image /data/datasets/fmp/imgs/${mode}/2019-07-22_2019-07-28_${mode}.nori.list \
    --max_images 1000 --max_epsilon ${eps} \
    --targeted \
    --speedup \
    --start ${start_id} \
    --cnt 50 \
    --mode ${mode} \
    --label ${label} &

    # mode='attack'
    # label=1
    # nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.square \
    # --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    # --image /data/datasets/fmp/imgs/${mode}/2019-07-22_2019-07-28_${mode}.nori.list \
    # --max_images 1000 --max_epsilon ${eps} \
    # --targeted \
    # --speedup \
    # --start ${start_id} \
    # --cnt 50 \
    # --mode ${mode} \
    # --label ${label} &


    # # 有目标攻击 normal
    # mode='normal'
    # label=0
    # nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.square \
    # --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    # --image /data/datasets/fmp/imgs/${mode}/2019-07-22_2019-07-28_${mode}.nori.list \
    # --max_images 1000 --max_epsilon ${eps} \
    # --targeted \
    # --speedup \
    # --start ${start_id} \
    # --cnt 50 \
    # --mode ${mode} \
    # --label ${label} &

    # mode='attack'
    # label=0
    # nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.square \
    # --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    # --image /data/datasets/fmp/imgs/${mode}/2019-07-22_2019-07-28_${mode}.nori.list \
    # --max_images 1000 --max_epsilon ${eps} \
    # --targeted \
    # --speedup \
    # --start ${start_id} \
    # --cnt 50 \
    # --mode ${mode} \
    # --label ${label} &
done