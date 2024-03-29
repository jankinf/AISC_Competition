# ckpt='/data/projects/CelebA-Spoof/ckpts/gpu8_bs1024_resnet18_sgd5e-3_wd5e-4_m9e-1_wo_cropped_jpeg_aug_corder_schedule15-30-45/14.pth.tar'
# ckpt='/data/projects/CelebA-Spoof/src/ckpt_iter.pth.tar'
ckpts=(
    '/data/projects/CelebA-Spoof/src/ckpt_iter.pth.tar'
    '/data/projects/CelebA-Spoof/ckpts/gpu8_bs1024_resnet18_sgd5e-3_wd5e-4_m9e-1_wo_cropped_jpeg_aug_corder_schedule15-30-45/14.pth.tar'
)


# file='out_png_2000'
# for ckpt in "${ckpts[@]}"
# do
#     nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.eval_adv \
#     --img_dir "/data/projects/CelebA-Spoof/attack/spoof_data/${file}" \
#     --log_file "/data/projects/CelebA-Spoof/attack/logs/baseline/${file}/log.log" \
#     --ckpt ${ckpt} &
# done
# file='out_png_pos'
# for ckpt in "${ckpts[@]}"
# do
#     nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.eval_adv \
#     --img_dir "/data/projects/CelebA-Spoof/attack/spoof_data/${file}" \
#     --log_file "/data/projects/CelebA-Spoof/attack/logs/baseline/${file}/log.log" \
#     --ckpt ${ckpt} &
# done
# file='out_png_neg'
# for ckpt in "${ckpts[@]}"
# do
#     nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m attack.eval_adv \
#     --img_dir "/data/projects/CelebA-Spoof/attack/spoof_data/${file}" \
#     --log_file "/data/projects/CelebA-Spoof/attack/logs/baseline/${file}/log.log" \
#     --ckpt ${ckpt} &
# done


file='out_png_pos'
for ckpt in "${ckpts[@]}"
do
    python -m attack.eval_adv \
    --img_dir "/data/projects/CelebA-Spoof/attack/spoof_data/${file}" \
    --log_file "/data/projects/CelebA-Spoof/attack/logs/baseline/${file}/log.log" \
    --ckpt ${ckpt}
done