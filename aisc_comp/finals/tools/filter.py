import os
import shutil
image_dir = "/data/projects/aisc_facecomp/finals/data/images"
paths = [
    "受害者1",
    "受害者2",
    "受害者3",
    "攻击者1",
    "攻击者2",
    "攻击者3",
]
ofs = 40
for name in paths:
    all_frames = os.listdir(os.path.join(image_dir, name))
    all_frames = sorted(all_frames, key=lambda x: int(os.path.splitext(x)[0]))
    frames = all_frames[::ofs]
    out_dir = "/data/projects/aisc_facecomp/finals/data/filter{}/{}".format(ofs, name)
    os.makedirs(out_dir, exist_ok=True)
    
    for frame in frames:
        src_path = os.path.join(image_dir, name, frame)
        tgt_path = os.path.join(out_dir, frame)
        shutil.copyfile(src_path, tgt_path)