import os
dir_name = "/data/projects/aisc_facecomp/raw_data/game3000"
list_ = os.listdir(dir_name)
for dirn in list_:
    if(len(os.listdir(os.path.join(dir_name, dirn))) <= 1):
        print(len(os.listdir(os.path.join(dir_name, dirn))), dirn)
