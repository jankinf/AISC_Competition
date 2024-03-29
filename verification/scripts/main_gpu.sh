# conda activate py36_megbrain

paths=(
    # /data/datasets/fmp/imgs/normal/2019-07-22_2019-07-28_normal.nori.list
    /data/projects/fmp_demo/attack/adv_data/preprocess224/normal
)
for path in "${paths[@]}"
do
    # nohup rlaunch --cpu 2 --memory 16394 -- python3 main_gpu.py \
    # --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    # --image $path \
    # --max_images 1000 &
    python3 main_batch.py \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --image $path \
    --max_images 1000
done