# conda activate py36_megbrain

epslist=(1.0 2.0 4.0 8.0 12.0 16.0)
methods=(i_fgsm_nt i_fgsm_t0 i_fgsm_t1)
for eps in "${epslist[@]}"
do
    for method in "${methods[@]}"
    do
        nohup rlaunch --cpu 2 --memory 16394 -- python3 main.py \
        --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
        --image /data/projects/fmp_demo/attack/adv_data/${method}/iter10_eps${eps}/attack \
        --logfile ${method}/iter10_eps${eps}/1000.log \
        --max_images 1000 &

        nohup rlaunch --cpu 2 --memory 16394 -- python3 main.py \
        --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
        --image /data/projects/fmp_demo/attack/adv_data/${method}/iter10_eps${eps}/normal \
        --logfile ${method}/iter10_eps${eps}/1000.log \
        --max_images 1000 &
    done
done