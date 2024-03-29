# ckpt='/data/projects/CelebA-Spoof/ckpts/gpu8_bs1024_resnet18_sgd5e-3_wd5e-4_m9e-1_wo_cropped_jpeg_aug_corder_schedule15-30-45/14.pth.tar'
ckpt='/data/projects/CelebA-Spoof/src/ckpt_iter.pth.tar'
python -m attack.methods.i_fgsm_t \
--max_epsilon 16 --num_iter 10 \
--workers 4 --batch_size 32 --target 0 \
--ckpt ${ckpt} \
--eval_ckpts \
'/data/projects/CelebA-Spoof/ckpts/gpu8_bs1024_resnet18_sgd5e-3_wd5e-4_m9e-1_wo_cropped_jpeg_aug_corder_schedule15-30-45/14.pth.tar' \
'/data/projects/CelebA-Spoof/src/ckpt_iter.pth.tar'
