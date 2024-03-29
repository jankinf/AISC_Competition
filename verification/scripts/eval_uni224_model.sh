# conda activate py36_megbrain

# paths=(
#     # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train900_iter10000_eps16/nq1821.0@suc900.npy
#     /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train900_iter10000_eps8/nq3034.0@suc867.npy
#     # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train900_iter10000_eps16/nq2942.0@suc645.npy
#     # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train900_iter10000_eps8/nq3690.0@suc643.npy
#     # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t1/train900_iter10000_eps16/nq1026.0@suc899.npy
#     # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t1/train900_iter10000_eps8/nq783.0@suc766.npy
#     # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t0/train900_iter10000_eps16/nq1717.0@suc649.npy
#     # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t0/train900_iter10000_eps8/nq1522.0@suc622.npy
# )
# train_img=900
# for path in "${paths[@]}"
# do
#     IFS='/' read -r -a array <<< $path
#     # python -m attack.eval_uni224 \
#     # --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
#     # --pattern $path \
#     # --name "${array[6]}/${array[7]}/${array[8]}" \
#     # --train_images ${train_img}
    
#     nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.eval_uni224 \
#     --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
#     --pattern $path \
#     --name "${array[6]}/${array[7]}/${array[8]}" \
#     --train_images ${train_img} &
# done


paths=(
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train900_iter10000_eps16/nq1821.0@suc900.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train900_iter10000_eps8/nq3034.0@suc867.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train900_iter10000_eps16/nq2942.0@suc645.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train900_iter10000_eps8/nq3690.0@suc643.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t1/train900_iter10000_eps16/nq1026.0@suc899.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t1/train900_iter10000_eps8/nq783.0@suc766.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t0/train900_iter10000_eps16/nq1717.0@suc649.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t0/train900_iter10000_eps8/nq1522.0@suc622.npy
    
    # # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train300_iter10000_eps16/nq9348.0@suc241.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train300_iter10000_eps8/nq9141.0@suc237.npy
    # # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train500_iter10000_eps16/nq6551.0@suc378.npy
    # # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train500_iter10000_eps4/nq6140.0@suc342.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train500_iter10000_eps8/nq5696.0@suc366.npy
    # # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train700_iter10000_eps16/nq1305.0@suc493.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train700_iter10000_eps8/nq4224.0@suc510.npy
    # # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train900_iter10000_eps16/nq2725.0@suc646.npy
    # # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train900_iter10000_eps8/nq3495.0@suc649.npy
    # # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train300_iter10000_eps16/nq1248.0@suc300.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train300_iter10000_eps8/nq4825.0@suc300.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train500_iter10000_eps10/nq3891.0@suc500.npy
    # # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train500_iter10000_eps12/nq869.0@suc486.npy
    # # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train500_iter10000_eps16/nq1596.0@suc500.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train500_iter10000_eps20/nq437.0@suc500.npy
    # # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train500_iter10000_eps4/nq2778.0@suc343.npy
    # # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train500_iter10000_eps6/nq1451.0@suc353.npy
    # # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train500_iter10000_eps8/nq8722.0@suc500.npy
    # # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train700_iter10000_eps16/nq1056.0@suc700.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train700_iter10000_eps8/nq2113.0@suc648.npy
    
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t0/train300_iter10000_eps16/nq4942.0@suc239.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t0/train300_iter10000_eps8/nq4581.0@suc227.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t0/train500_iter10000_eps16/nq2896.0@suc368.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t0/train500_iter10000_eps4/nq2898.0@suc331.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t0/train500_iter10000_eps8/nq1776.0@suc334.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t0/train700_iter10000_eps16/nq1567.0@suc488.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t0/train700_iter10000_eps8/nq1131.0@suc476.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t0/train900_iter10000_eps8/nq1299.0@suc628.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t1/train300_iter10000_eps16/nq554.0@suc299.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t1/train300_iter10000_eps8/nq5326.0@suc299.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t1/train500_iter10000_eps10/nq207.0@suc395.npy

    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t1/train500_iter10000_eps12/nq118.0@suc403.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t1/train500_iter10000_eps16/nq115.0@suc432.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t1/train500_iter10000_eps20/nq101.0@suc431.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t1/train500_iter10000_eps4/nq3611.0@suc342.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t1/train500_iter10000_eps6/nq81.0@suc284.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t1/train500_iter10000_eps8/nq221.0@suc353.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t1/train700_iter10000_eps16/nq558.0@suc699.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_simba_hardupdate_t1/train700_iter10000_eps8/nq1396.0@suc634.npy


    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train100_iter10000_eps3/nq6976.0@suc75.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train300_iter10000_eps3/nq9988.0@suc212.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train400_iter10000_eps3/nq8904.0@suc273.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train50_iter10000_eps3/nq9998.0@suc45.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train100_iter10000_eps3/nq9533.0@suc81.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train300_iter10000_eps3/nq9744.0@suc206.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train200_iter10000_eps3/nq8679.0@suc147.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train50_iter10000_eps3/nq9985.0@suc37.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train400_iter10000_eps3/nq9893.0@suc275.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train200_iter10000_eps3/nq9999.0@suc146.npy

    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train500_iter10000_eps4/nq6140.0@suc342.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train50_iter10000_eps4/nq9984.0@suc40.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train200_iter10000_eps4/nq9992.0@suc151.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train100_iter10000_eps4/nq9825.0@suc78.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train400_iter10000_eps4/nq9876.0@suc288.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train300_iter10000_eps4/nq9993.0@suc218.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train500_iter10000_eps4/nq2778.0@suc343.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train50_iter10000_eps4/nq9999.0@suc47.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train100_iter10000_eps4/nq9999.0@suc89.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train400_iter10000_eps4/nq9812.0@suc301.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train200_iter10000_eps4/nq9989.0@suc158.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train300_iter10000_eps4/nq9997.0@suc223.npy

    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train100_iter10000_eps5/nq9172.0@suc80.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train300_iter10000_eps5/nq9885.0@suc227.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train50_iter10000_eps5/nq9995.0@suc41.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train400_iter10000_eps5/nq9960.0@suc292.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train200_iter10000_eps5/nq7942.0@suc150.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train300_iter10000_eps5/nq9933.0@suc239.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train100_iter10000_eps5/nq9990.0@suc96.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train50_iter10000_eps5/nq4351.0@suc44.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train400_iter10000_eps5/nq9917.0@suc326.npy
    # /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train200_iter10000_eps5/nq9988.0@suc173.npy

    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train300_iter10000_eps6/nq9953.0@suc237.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train50_iter10000_eps6/nq9832.0@suc41.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train100_iter10000_eps6/nq9994.0@suc82.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train200_iter10000_eps6/nq9959.0@suc162.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t0/train400_iter10000_eps6/nq9932.0@suc297.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train500_iter10000_eps6/nq1451.0@suc353.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train100_iter10000_eps6/nq7716.0@suc100.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train50_iter10000_eps6/nq8161.0@suc50.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train300_iter10000_eps6/nq9999.0@suc294.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train400_iter10000_eps6/nq10000.0@suc388.npy
    /data/projects/fmp_demo/attack/uniscore_noise/uniscore_square_minloss_t1/train200_iter10000_eps6/nq9626.0@suc194.npy
)

for path in "${paths[@]}"
do
    IFS='/' read -r -a array <<< $path
    # python -m attack.eval_uni224 \
    # --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    # --pattern $path \
    # --name "${array[6]}/${array[7]}/${array[8]}" \
    # --train_images ${train_img}
    
    nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.eval_uni224 \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --pattern $path \
    --name "${array[6]}/${array[7]}/${array[8]}" &
done
