
paths=(
    /data/projects/fmp_demo/attack/adv_data/preprocess224
)
for path in "${paths[@]}"
do
    nohup rlaunch --cpu 1 --memory 8394 -- python -m attack.eval224 \
    --i_net panorama.i_epoch_208528.th_0.0162.neupeak \
    --data_dir $path \
    --name "blackbox_eval/baseline_process224" &
done
