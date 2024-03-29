# python -m aisc.attacks.main \
# --steps 200 \
# --alpha 0.0005 \
# --n_ens 3 \
# --resize_rate 1.3 \
# --diversity_prob 1.0 \
# --model_idx 0 1 7 8 15

# python -m aisc.attacks.main_encode \
# --steps 200 \
# --alpha 0.001 \
# --n_ens 2 \
# --resize_rate 1.3 \
# --diversity_prob 1.0 \
# --model_idx 0 1 3 4 7 8 13 15

python -m aisc.attacks.main_encode_sproof \
--steps 200 \
--alpha 0.001 \
--n_ens 2 \
--resize_rate 1.3 \
--diversity_prob 1.0 \
--model_idx 0 1 3 4 7 8 13 15

# python -m aisc.attacks.main_grad \
# --steps 400 \
# --alpha 0.3 \
# --n_ens 3 \
# --resize_rate 1.3 \
# --diversity_prob 1.0 \
# --model_idx 0 1 7 8