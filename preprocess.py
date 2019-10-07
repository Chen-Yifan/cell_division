import os,re
from shutil import copyfile
import numpy as np
orig_path = '/home/yifanc3/dataset2/cell_dataset/cells'
frame_path = '/home/yifanc3/dataset2/cell_dataset/frames_all'
mask_path = '/home/yifanc3/dataset2/cell_dataset/masks_all'
def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def split_label_and_RGB():
    mkdir(frame_path)
    mkdir(mask_path)
    files = os.listdir(orig_path)
    files.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                for x in re.findall(r'[^0-9]|[0-9]+', var)])
    for i in range(len(files)):
        print(files[i])
        file = files[i]
        src = os.path.join(orig_path, file)

        if i%2==0: # R < l
            # frame
            dst = os.path.join(frame_path, file)
            copyfile(src, dst)
        else:
            dst = os.path.join(mask_path, file)
            copyfile(src, dst)
            
            

if __name__ == '__main__':
    __3_channel_to_1()
