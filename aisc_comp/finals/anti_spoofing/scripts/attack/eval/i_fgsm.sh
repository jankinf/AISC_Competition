# ckpt='/data/projects/CelebA-Spoof/ckpts/gpu8_bs1024_resnet18_sgd5e-3_wd5e-4_m9e-1_wo_cropped_jpeg_aug_corder_schedule15-30-45/14.pth.tar'
ckpt='/data/projects/CelebA-Spoof/src/ckpt_iter.pth.tar'

method='i_fgsm'
setting='iter10_eps0.1'

python -m attack.eval_adv \
--img_dir "/data/projects/CelebA-Spoof/attack/adv_data/${method}/${setting}" \
--log_file "/data/projects/CelebA-Spoof/attack/logs/${method}/${setting}/log.log" \
--ckpt ${ckpt}

# python -m attack.methods.i_fgsm \
# --max_epsilon 16 --num_iter 10 \
# --workers 4 --batch_size 32 \
# --ckpt ${ckpt}


# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.i_fgsm \
# --max_epsilon 16 --num_iter 10 \
# --workers 4 --batch_size 32 \
# --ckpt ${ckpt} > /data/projects/CelebA-Spoof/logs/attacks/ 2>&1 &

# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.i_fgsm \
# --max_epsilon 14 --num_iter 10 \
# --workers 4 --batch_size 32 \
# --ckpt ${ckpt} &

# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.i_fgsm \
# --max_epsilon 12 --num_iter 10 \
# --workers 4 --batch_size 32 \
# --ckpt ${ckpt} &

# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.i_fgsm \
# --max_epsilon 10 --num_iter 10 \
# --workers 4 --batch_size 32 \
# --ckpt ${ckpt} &

# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.i_fgsm \
# --max_epsilon 8 --num_iter 10 \
# --workers 4 --batch_size 32 \
# --ckpt ${ckpt} &

# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.i_fgsm \
# --max_epsilon 6 --num_iter 10 \
# --workers 4 --batch_size 32 \
# --ckpt ${ckpt} &

# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.i_fgsm \
# --max_epsilon 4 --num_iter 10 \
# --workers 4 --batch_size 32 \
# --ckpt ${ckpt} &

# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.i_fgsm \
# --max_epsilon 2 --num_iter 10 \
# --workers 4 --batch_size 32 \
# --ckpt ${ckpt} &

# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.i_fgsm \
# --max_epsilon 1 --num_iter 10 \
# --workers 4 --batch_size 32 \
# --ckpt ${ckpt} &

# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.i_fgsm \
# --max_epsilon 0.5 --num_iter 10 \
# --workers 4 --batch_size 32 \
# --ckpt ${ckpt} &

# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.i_fgsm \
# --max_epsilon 0.1 --num_iter 10 \
# --workers 4 --batch_size 32 \
# --ckpt ${ckpt} &

# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.i_fgsm \
# --max_epsilon 0.2 --num_iter 10 \
# --workers 4 --batch_size 32 \
# --ckpt ${ckpt} &
# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.i_fgsm \
# --max_epsilon 0.3 --num_iter 10 \
# --workers 4 --batch_size 32 \
# --ckpt ${ckpt} &
# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.i_fgsm \
# --max_epsilon 0.4 --num_iter 10 \
# --workers 4 --batch_size 32 \
# --ckpt ${ckpt} &

# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.i_fgsm \
# --max_epsilon 0.05 --num_iter 10 \
# --workers 4 --batch_size 32 \
# --ckpt ${ckpt} &
# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.i_fgsm \
# --max_epsilon 0.01 --num_iter 10 \
# --workers 4 --batch_size 32 \
# --ckpt ${ckpt} &
# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.methods.i_fgsm \
# --max_epsilon 0.08 --num_iter 10 \
# --workers 4 --batch_size 32 \
# --ckpt ${ckpt} &
