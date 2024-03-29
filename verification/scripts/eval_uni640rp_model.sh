
paths=(    
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train300_iter10000_eps8/nq9141.0@suc237.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train500_iter10000_eps8/nq5696.0@suc366.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train300_iter10000_eps8/nq4825.0@suc300.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train500_iter10000_eps8/nq8722.0@suc500.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train900_iter10000_eps8/nq3034.0@suc867.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train700_iter10000_eps8/nq4224.0@suc510.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train700_iter10000_eps8/nq2113.0@suc648.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train900_iter10000_eps8/nq3495.0@suc649.npy
    
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train300_iter10000_eps16/nq9348.0@suc241.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train700_iter10000_eps16/nq1305.0@suc493.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train500_iter10000_eps16/nq6551.0@suc378.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train900_iter10000_eps16/nq2725.0@suc646.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train300_iter10000_eps16/nq1248.0@suc300.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train500_iter10000_eps16/nq1596.0@suc500.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train700_iter10000_eps16/nq1056.0@suc700.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train900_iter10000_eps16/nq1821.0@suc900.npy
)

for path in "${paths[@]}"
do
    IFS='/' read -r -a array <<< $path
    nohup rlaunch --cpu 2 --memory 26394 --gpu 1 -- python -m attack.eval_uni640rp \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --pattern $path \
    --name "${array[6]}/${array[7]}/${array[8]}" > debug.out &
    # python -m attack.eval_uni640rp \
    # --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    # --pattern $path \
    # --name "${array[6]}/${array[7]}/${array[8]}"
done
