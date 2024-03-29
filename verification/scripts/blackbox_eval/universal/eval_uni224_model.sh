# conda activate py36_megbrain

paths=(
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train200_iter10000_eps3/nq8679.0@suc147.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train200_iter10000_eps3/nq9999.0@suc146.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train200_iter10000_eps4/nq9992.0@suc151.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train200_iter10000_eps4/nq9989.0@suc158.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train200_iter10000_eps5/nq7942.0@suc150.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train200_iter10000_eps5/nq9988.0@suc173.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train200_iter10000_eps6/nq9959.0@suc162.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train200_iter10000_eps6/nq9626.0@suc194.npy  

    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train300_iter10000_eps3/nq9988.0@suc212.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train300_iter10000_eps3/nq9744.0@suc206.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train300_iter10000_eps4/nq9993.0@suc218.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train300_iter10000_eps4/nq9997.0@suc223.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train300_iter10000_eps5/nq9885.0@suc227.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train300_iter10000_eps5/nq9933.0@suc239.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train300_iter10000_eps6/nq9953.0@suc237.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train300_iter10000_eps6/nq9999.0@suc294.npy

    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train400_iter10000_eps3/nq8904.0@suc273.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train400_iter10000_eps3/nq9893.0@suc275.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train400_iter10000_eps4/nq9876.0@suc288.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train400_iter10000_eps4/nq9812.0@suc301.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train400_iter10000_eps5/nq9960.0@suc292.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train400_iter10000_eps5/nq9917.0@suc326.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train400_iter10000_eps6/nq9932.0@suc297.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train400_iter10000_eps6/nq10000.0@suc388.npy
)

for path in "${paths[@]}"
do
    IFS='/' read -r -a array <<< $path
    nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.eval_uni224 \
    --i_net panorama.i_epoch_208528.th_0.0162.neupeak \
    --pattern $path \
    --name "blackbox_eval/uni@query/eval_uni224/${array[6]}/${array[7]}/${array[8]}" &
done
