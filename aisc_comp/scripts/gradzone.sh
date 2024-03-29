# python gradzone.py \
# --output_dir ./gradzone/ds_ens1_model0_iter5_alpha1e-6 \
# --batch_size 1 \
# --alpha 1.6 \
# --steps 5 \
# --model_idx 0 \
# --device 0 

step=10
alpha=1.6
nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python gradzone.py \
--output_dir ./gradzone/ds_ens4_model0178_iter${step}_alpha1e-6 \
--batch_size 1 \
--alpha 1.6 \
--steps $step \
--model_idx 0 1 7 8 \
--device 0 &

sleep 5s

step=10
alpha=1.6
nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python gradzone.py \
--output_dir ./gradzone/ds_ens10_model0123456789_iter${step}_alpha1e-6 \
--batch_size 1 \
--alpha 1.6 \
--steps $step \
--model_idx 0 1 2 3 4 5 6 7 8 9 \
--device 0 &