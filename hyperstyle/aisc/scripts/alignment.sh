paths=(
    受害者1
    受害者2
    受害者3
    攻击者1
    攻击者2
    攻击者3
)

for path in ${paths[@]}
do
    python -m aisc.alignment \
    --input_dir /data/projects/aisc_facecomp/finals/data/filter40/${path} \
    --output_dir /data/projects/hyperstyle/aisc/aligned1024/${path}
done