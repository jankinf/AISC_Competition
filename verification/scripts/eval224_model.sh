# # conda activate py36_megbrain
# paths=(
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps1
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps10
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps12
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps14
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps16
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps2
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps3
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps4
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps5
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps6
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps7
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps8
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps1
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps10
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps12
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps14
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps16
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps2
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps3
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps4
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps5
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps6
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps7
#     /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps8
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps1
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps10
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps12
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps14
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps16
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps2
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps3
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps4
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps5
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps6
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps7
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps8
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps1
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps10
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps12
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps14
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps16
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps2
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps3
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps4
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps5
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps6
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps7
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps8
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps1
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps10
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps12
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps14
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps16
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps2
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps3
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps4
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps5
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps6
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps7
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps8
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps1
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps10
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps12
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps14
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps16
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps2
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps3
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps4
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps5
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps6
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps7
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps8
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps1
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps10
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps12
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps14
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps16
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps2
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps3
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps4
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps5
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps6
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps7
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps8
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps1
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps10
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps12
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps14
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps16
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps2
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps3
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps4
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps5
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps6
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps7
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps8
# )

# for path in "${paths[@]}"
# do
#     IFS='/' read -r -a array <<< $path
#     # python -m attack.eval224 \
#     # --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
#     # --data_dir $path \
#     # --name "${array[6]}/${array[7]}/${array[8]}"
    
#     nohup rlaunch --cpu 1 --memory 8394 -- python -m attack.eval224 \
#     --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
#     --data_dir $path \
#     --name "${array[6]}/${array[7]}/${array[8]}" &
# done

# wait

# paths=(
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps1
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps10
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps12
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps14
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps16
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps2
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps3
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps4
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps5
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps6
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps7
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t0/iter10_eps8
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps1
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps10
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps12
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps14
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps16
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps2
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps3
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps4
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps5
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps6
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps7
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/i_fgsm_t1/iter10_eps8
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps1
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps10
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps12
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps14
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps16
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps2
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps3
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps4
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps5
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps6
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps7
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t0/iter10_eps8
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps1
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps10
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps12
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps14
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps16
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps2
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps3
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps4
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps5
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps6
#     # /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps7
#     /data/projects/fmp_demo/attack/adv_data/whitebox/mi_fgsm_t1/iter10_eps8
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps1
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps10
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps12
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps14
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps16
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps2
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps3
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps4
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps5
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps6
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps7
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps8
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps1
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps10
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps12
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps14
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps16
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps2
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps3
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps4
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps5
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps6
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps7
#     /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t1/iter10_eps8
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps1
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps10
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps12
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps14
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps16
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps2
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps3
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps4
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps5
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps6
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps7
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t0/iter10_eps8
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps1
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps10
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps12
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps14
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps16
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps2
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps3
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps4
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps5
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps6
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps7
#     /data/projects/fmp_demo/attack/adv_data/whitebox/smi_fgsm_t1/iter10_eps8
# )

# for path in "${paths[@]}"
# do
#     IFS='/' read -r -a array <<< $path
#     # python -m attack.eval224 \
#     # --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
#     # --data_dir $path \
#     # --name "${array[6]}/${array[7]}/${array[8]}"
    
#     nohup rlaunch --cpu 1 --memory 8394 -- python -m attack.eval224 \
#     --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
#     --data_dir $path \
#     --name "${array[6]}/${array[7]}/${array[8]}" &
# done


paths=(
    /data/projects/fmp_demo/attack/adv_data/whitebox/si_fgsm_t0/iter10_eps16
)
for path in "${paths[@]}"
do
    IFS='/' read -r -a array <<< $path
    # python -m attack.eval224 \
    # --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    # --data_dir $path \
    # --name "${array[6]}/${array[7]}/${array[8]}"
    
    nohup rlaunch --cpu 1 --memory 8394 -- python -m attack.eval224 \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --data_dir $path \
    --name "${array[6]}/${array[7]}/${array[8]}" &
done
