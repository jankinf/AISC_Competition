# conda activate py36_megbrain

paths=(
    # ./attack/adv_data/ct_fgsm_nt/iter10_eps16.0/attack
    # ./attack/adv_data/ct_fgsm_nt/iter10_eps16.0/normal
    # ./attack/adv_data/ct_fgsm_t1/iter10_eps16.0/attack
    # ./attack/adv_data/ct_fgsm_t1/iter10_eps16.0/normal
    # ./attack/adv_data/ct_fgsm_nt/iter10_eps1.0/attack
    # ./attack/adv_data/ct_fgsm_nt/iter10_eps1.0/normal
    # ./attack/adv_data/ct_fgsm_t1/iter10_eps1.0/attack
    # ./attack/adv_data/ct_fgsm_t1/iter10_eps1.0/normal
    # ./attack/adv_data/ct_fgsm_nt/iter10_eps8.0/attack
    # ./attack/adv_data/ct_fgsm_nt/iter10_eps8.0/normal
    # ./attack/adv_data/ct_fgsm_t1/iter10_eps8.0/attack
    # ./attack/adv_data/ct_fgsm_t1/iter10_eps8.0/normal
    # ./attack/adv_data/ct_fgsm_t0/iter10_eps8.0/attack
    # ./attack/adv_data/ct_fgsm_t0/iter10_eps8.0/normal
    # ./attack/adv_data/ct_fgsm_t0/iter10_eps16.0/attack
    # ./attack/adv_data/ct_fgsm_t0/iter10_eps16.0/normal
    # ./attack/adv_data/ct_fgsm_t0/iter10_eps1.0/attack
    # ./attack/adv_data/ct_fgsm_t0/iter10_eps1.0/normal
    # attack/adv_data/square_nt/iter5000_eps4/attack
    # attack/adv_data/square_nt/iter5000_eps4/normal
    # attack/adv_data/square_t0/iter5000_eps4/attack
    # attack/adv_data/square_t0/iter5000_eps4/normal
    # attack/adv_data/square_t1/iter5000_eps4/attack
    # attack/adv_data/square_t1/iter5000_eps4/normal

#     attack/adv_data/square_t1_restart2/iter5000_eps4/attack
#     attack/adv_data/square_t1/iter5000_eps1/attack
#     attack/adv_data/square_t1/iter5000_eps2/attack
#     attack/adv_data/square_t1/iter5000_eps3/attack
#     attack/adv_data/square_t1/iter5000_eps4/attack
#     attack/adv_data/square_t1/iter5000_eps5/attack
#     attack/adv_data/square_t1/iter5000_eps6/attack
#     attack/adv_data/square_t1/iter5000_eps7/attack
#     attack/adv_data/square_t1/iter5000_eps8/attack
    attack/adv_data/square_t1/iter5000_eps1/normal
#     attack/adv_data/square_t1/iter5000_eps2/normal
#     attack/adv_data/square_t1/iter5000_eps3/normal
#     attack/adv_data/square_t1/iter5000_eps4/normal
#     attack/adv_data/square_t1/iter5000_eps5/normal
#     attack/adv_data/square_t1/iter5000_eps6/normal
#     attack/adv_data/square_t1/iter5000_eps7/normal
#     attack/adv_data/square_t1/iter5000_eps8/normal

#     attack/adv_data/square_t0/iter5000_eps1/attack
#     attack/adv_data/square_t0/iter5000_eps2/attack
#     attack/adv_data/square_t0/iter5000_eps3/attack
#     attack/adv_data/square_t0/iter5000_eps4/attack
#     attack/adv_data/square_t0/iter5000_eps5/attack
#     attack/adv_data/square_t0/iter5000_eps6/attack
#     attack/adv_data/square_t0/iter5000_eps7/attack
#     attack/adv_data/square_t0/iter5000_eps8/attack
#     attack/adv_data/square_t0/iter5000_eps1/normal
#     attack/adv_data/square_t0/iter5000_eps2/normal
#     attack/adv_data/square_t0/iter5000_eps3/normal
#     attack/adv_data/square_t0/iter5000_eps4/normal
#     attack/adv_data/square_t0/iter5000_eps5/normal
#     attack/adv_data/square_t0/iter5000_eps6/normal
#     attack/adv_data/square_t0/iter5000_eps7/normal
#     attack/adv_data/square_t0/iter5000_eps8/normal
)
for path in "${paths[@]}"
do
    IFS='/' read -r -a array <<< $path
    nohup rlaunch --cpu 2 --memory 16394 -- python3 main.py \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --image $path \
    --logfile "${array[2]}/${array[3]}/1000.log" \
    --max_images 1000 &
done