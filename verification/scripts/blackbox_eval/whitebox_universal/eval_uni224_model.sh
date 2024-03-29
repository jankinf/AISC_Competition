# conda activate py36_megbrain

paths=(
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train1000_bs100_epoch50_eps10_step0.6_celoss_beta12.0/epoch49@suc911.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train1000_bs100_epoch50_eps3_step0.6_celoss_beta12.0/epoch49@suc707.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train1000_bs100_epoch50_eps4_step0.6_celoss_beta12.0/epoch49@suc699.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train1000_bs100_epoch50_eps5_step0.6_celoss_beta12.0/epoch49@suc704.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train1000_bs100_epoch50_eps6_step0.6_celoss_beta12.0/epoch49@suc737.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train1000_bs100_epoch50_eps8_step0.6_celoss_beta12.0/epoch49@suc841.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train1000_bs50_epoch50_eps10_step0.6_celoss_beta12.0/epoch49@suc805.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train1000_bs50_epoch50_eps3_step0.6_celoss_beta12.0/epoch49@suc628.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train1000_bs50_epoch50_eps4_step0.6_celoss_beta12.0/epoch49@suc654.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train1000_bs50_epoch50_eps5_step0.6_celoss_beta12.0/epoch49@suc643.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train1000_bs50_epoch50_eps6_step0.6_celoss_beta12.0/epoch49@suc692.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train1000_bs50_epoch50_eps8_step0.6_celoss_beta12.0/epoch49@suc658.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train300_bs100_epoch50_eps10_step0.6_celoss_beta12.0/epoch49@suc294.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train300_bs100_epoch50_eps3_step0.6_celoss_beta12.0/epoch49@suc221.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train300_bs100_epoch50_eps4_step0.6_celoss_beta12.0/epoch49@suc225.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train300_bs100_epoch50_eps5_step0.6_celoss_beta12.0/epoch49@suc231.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train300_bs100_epoch50_eps6_step0.6_celoss_beta12.0/epoch49@suc242.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train300_bs100_epoch50_eps8_step0.6_celoss_beta12.0/epoch49@suc259.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train300_bs50_epoch50_eps10_step0.6_celoss_beta12.0/epoch49@suc253.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train300_bs50_epoch50_eps3_step0.6_celoss_beta12.0/epoch49@suc210.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train300_bs50_epoch50_eps4_step0.6_celoss_beta12.0/epoch49@suc217.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train300_bs50_epoch50_eps5_step0.6_celoss_beta12.0/epoch49@suc211.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train300_bs50_epoch50_eps6_step0.6_celoss_beta12.0/epoch49@suc218.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train300_bs50_epoch50_eps8_step0.6_celoss_beta12.0/epoch49@suc220.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train500_bs100_epoch50_eps10_step0.6_celoss_beta12.0/epoch49@suc463.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train500_bs100_epoch50_eps3_step0.6_celoss_beta12.0/epoch49@suc352.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train500_bs100_epoch50_eps4_step0.6_celoss_beta12.0/epoch49@suc357.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train500_bs100_epoch50_eps5_step0.6_celoss_beta12.0/epoch49@suc363.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train500_bs100_epoch50_eps6_step0.6_celoss_beta12.0/epoch49@suc383.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train500_bs100_epoch50_eps8_step0.6_celoss_beta12.0/epoch49@suc428.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train500_bs50_epoch50_eps10_step0.6_celoss_beta12.0/epoch49@suc422.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train500_bs50_epoch50_eps3_step0.6_celoss_beta12.0/epoch49@suc332.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train500_bs50_epoch50_eps4_step0.6_celoss_beta12.0/epoch49@suc328.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train500_bs50_epoch50_eps5_step0.6_celoss_beta12.0/epoch49@suc332.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train500_bs50_epoch50_eps6_step0.6_celoss_beta12.0/epoch49@suc346.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t0/train500_bs50_epoch50_eps8_step0.6_celoss_beta12.0/epoch49@suc371.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train1000_bs100_epoch50_eps10_step0.6_celoss_beta12.0/epoch49@suc952.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train1000_bs100_epoch50_eps3_step0.6_celoss_beta12.0/epoch49@suc855.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train1000_bs100_epoch50_eps4_step0.6_celoss_beta12.0/epoch49@suc944.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train1000_bs100_epoch50_eps6_step0.6_celoss_beta12.0/epoch49@suc985.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train1000_bs100_epoch50_eps8_step0.6_celoss_beta12.0/epoch49@suc959.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train1000_bs50_epoch50_eps10_step0.6_celoss_beta12.0/epoch49@suc794.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train1000_bs50_epoch50_eps3_step0.6_celoss_beta12.0/epoch49@suc702.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train1000_bs50_epoch50_eps4_step0.6_celoss_beta12.0/epoch49@suc866.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train1000_bs50_epoch50_eps5_step0.6_celoss_beta12.0/epoch49@suc930.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train1000_bs50_epoch50_eps6_step0.6_celoss_beta12.0/epoch49@suc945.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train1000_bs50_epoch50_eps8_step0.6_celoss_beta12.0/epoch49@suc955.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train300_bs100_epoch50_eps10_step0.6_celoss_beta12.0/epoch49@suc294.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train300_bs100_epoch50_eps3_step0.6_celoss_beta12.0/epoch49@suc256.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train300_bs100_epoch50_eps4_step0.6_celoss_beta12.0/epoch49@suc285.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train300_bs100_epoch50_eps5_step0.6_celoss_beta12.0/epoch49@suc294.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train300_bs100_epoch50_eps6_step0.6_celoss_beta12.0/epoch49@suc296.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train300_bs100_epoch50_eps8_step0.6_celoss_beta12.0/epoch49@suc300.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train300_bs50_epoch50_eps10_step0.6_celoss_beta12.0/epoch49@suc299.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train300_bs50_epoch50_eps3_step0.6_celoss_beta12.0/epoch49@suc231.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train300_bs50_epoch50_eps4_step0.6_celoss_beta12.0/epoch49@suc277.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train300_bs50_epoch50_eps5_step0.6_celoss_beta12.0/epoch49@suc287.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train300_bs50_epoch50_eps6_step0.6_celoss_beta12.0/epoch49@suc283.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train300_bs50_epoch50_eps8_step0.6_celoss_beta12.0/epoch49@suc282.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train500_bs100_epoch50_eps10_step0.6_celoss_beta12.0/epoch49@suc497.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train500_bs100_epoch50_eps3_step0.6_celoss_beta12.0/epoch49@suc446.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train500_bs100_epoch50_eps4_step0.6_celoss_beta12.0/epoch49@suc426.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train500_bs100_epoch50_eps5_step0.6_celoss_beta12.0/epoch49@suc495.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train500_bs100_epoch50_eps6_step0.6_celoss_beta12.0/epoch49@suc495.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train500_bs100_epoch50_eps8_step0.6_celoss_beta12.0/epoch49@suc495.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train500_bs50_epoch50_eps10_step0.6_celoss_beta12.0/epoch49@suc497.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train500_bs50_epoch50_eps3_step0.6_celoss_beta12.0/epoch49@suc356.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train500_bs50_epoch50_eps4_step0.6_celoss_beta12.0/epoch49@suc352.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train500_bs50_epoch50_eps5_step0.6_celoss_beta12.0/epoch49@suc473.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train500_bs50_epoch50_eps6_step0.6_celoss_beta12.0/epoch49@suc489.npy
    /data/projects/fmp_demo/attack/whitebox/uap_sgd_t1/train500_bs50_epoch50_eps8_step0.6_celoss_beta12.0/epoch49@suc492.npy

)

for path in "${paths[@]}"
do
    IFS='/' read -r -a array <<< $path
    nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.eval_uni224 \
    --i_net panorama.i_epoch_208528.th_0.0162.neupeak \
    --pattern $path \
    --name "blackbox_eval/uni@whitebox/eval_uni224/${array[6]}/${array[7]}/${array[8]}" &
done
