
paths=(    

    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t0/train200_iter10000_eps3/nq1247.0@suc130.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t0/train400_iter10000_eps3/nq2095.0@suc254.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t0/train500_iter10000_eps3/nq1910.0@suc308.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t0/train300_iter10000_eps3/nq2895.0@suc195.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t1/train500_iter10000_eps3/nq1221.0@suc236.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t1/train400_iter10000_eps3/nq2210.0@suc189.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t1/train300_iter10000_eps3/nq2316.0@suc145.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t1/train200_iter10000_eps3/nq2101.0@suc98.npy


    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t0/train200_iter10000_eps4/nq7008.0@suc157.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t0/train300_iter10000_eps4/nq5644.0@suc209.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t0/train400_iter10000_eps4/nq4568.0@suc275.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t0/train500_iter10000_eps4/nq2853.0@suc331.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t1/train200_iter10000_eps4/nq3.0@suc97.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t1/train400_iter10000_eps4/nq1811.0@suc190.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t1/train300_iter10000_eps4/nq3.0@suc137.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t1/train500_iter10000_eps4/nq1795.0@suc237.npy


    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t0/train300_iter10000_eps5/nq5248.0@suc211.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t0/train500_iter10000_eps5/nq2474.0@suc342.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t0/train400_iter10000_eps5/nq4757.0@suc288.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t0/train200_iter10000_eps5/nq5194.0@suc156.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t1/train500_iter10000_eps5/nq588.0@suc236.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t1/train400_iter10000_eps5/nq639.0@suc189.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t1/train300_iter10000_eps5/nq1@suc142.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t1/train200_iter10000_eps5/nq1@suc96.npy



    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t0/train300_iter10000_eps6/nq5937.0@suc226.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t0/train500_iter10000_eps6/nq3156.0@suc350.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t0/train400_iter10000_eps6/nq3222.0@suc285.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t0/train200_iter10000_eps6/nq7177.0@suc164.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t1/train500_iter10000_eps6/nq609.0@suc244.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t1/train200_iter10000_eps6/nq8.0@suc94.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t1/train300_iter10000_eps6/nq2726.0@suc154.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_640@480_t1/train400_iter10000_eps6/nq2350.0@suc209.npy

)

for path in "${paths[@]}"
do
    IFS='/' read -r -a array <<< $path
    nohup rlaunch --cpu 1 --memory 6394 -- python -m attack.eval_uni640_detector+model \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --pattern $path \
    --name "${array[6]}/${array[7]}/${array[8]}" > eval_uni640_detector+model.out &
    # python -m attack.eval_uni640_detector+model \
    # --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    # --pattern $path \
    # --name "${array[6]}/${array[7]}/${array[8]}"
done
