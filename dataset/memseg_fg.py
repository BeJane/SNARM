import glob
import os

import numpy as np

import cv2
def generate_target_foreground_mask(img: np.ndarray,bg_reverse=False,bg_threshold=127) -> np.ndarray:
    # convert RGB into GRAY scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray/255
    # generate binary mask of gray scale image
    # _, target_background_mask = cv2.threshold(img_gray,bg_threshold, 255, cv2.THRESH_BINARY)
    # target_background_mask = target_background_mask.astype(np.bool).astype(np.int)
    #
    # # invert mask for foreground mask
    # if bg_reverse:
    #     target_foreground_mask = target_background_mask
    # else:
    #     target_foreground_mask = -(target_background_mask - 1)
    #
    # return target_foreground_mask

outdir = '/media/szcyxy/新加卷/MemSeg_foreground'
datadir = '../data/defect_512/mvtec'

for c in os.listdir(outdir):
    pathlist = sorted([*glob.glob(os.path.join(datadir, c,'train','ok', '*.png'))])

    for path in pathlist:
        img = cv2.imread(path)
        mask = generate_target_foreground_mask(img,bg_reverse=False,bg_threshold=50)

        np.save(os.path.join(outdir,c,os.path.basename(path).split('.')[0]),mask)