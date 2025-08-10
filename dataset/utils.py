import math
import os.path

import cv2
import sympy as sympy
import torch
import torch.nn as nn
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from tqdm import tqdm


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"

    xy = x.dot(y)
    x2y2 = np.linalg.norm(x, ord=2) * np.linalg.norm(x, ord=2)
    sim = xy/x2y2
    return sim
def plt_position_code(pos_embed,h,w,save_path):
    l = h*w
    cos = np.zeros((l,l))
    # 计算所有
    for i in range(l):
        for j in range(l):
            cos[i, j] = cosine_similarity(pos_embed[i], pos_embed[j])

    fig, axs = plt.subplots(nrows=h, ncols=w, figsize=(h+3, w),
                            subplot_kw={'xticks': [], 'yticks': []})
    fig.suptitle(f'{os.path.basename(save_path).split(".")[0]}')
    i = 0
    cos = cos.reshape(l, h, w)
    for ax in axs.flat:
        ax.imshow(cos[i, :, :], cmap='viridis')
        i += 1
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)

def plt_hist(pos_embed,h,w,save_path):
    l = h*w
    # cos = np.zeros((l,l))
    # 计算所有
    # for i in range(l):
    #     for j in range(l):
    #         cos[i, j] = cosine_similarity(pos_embed[i], pos_embed[j])
    #
    fig, axs = plt.subplots(nrows=h, ncols=w, figsize=(h+3, w),
                            subplot_kw={'xticks': [], 'yticks': []})
    fig.suptitle(f'{os.path.basename(save_path).replace(".png","")}')
    i = 0
    # cos = cos.reshape(l, h, w)
    for ax in tqdm(axs.flat):
        # ax.imshow(cos[i, :, :], cmap='viridis')
        ax.bar(np.arange(0, pos_embed[i].shape[0]),pos_embed[i])
        i += 1
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)
def get_pad_color(image: np.ndarray):
    t1 = image[0, :]
    t2 = image[-1, :]
    t3 = image[:, 0]
    t4 = image[:, -1]
    t = np.concatenate([t1, t2, t3, t4], axis=0)
    return np.median(t, axis=0).tolist()

def filter_file_name(file_name: str):
    return file_name.replace(' ', '')  # 去除空格
# 加载指定文件夹下的图片
def load_img(img_path_list):
    img_list, img_name_list = [], []
    # print(img_path_list[0])
    # if len(img_path_list) > 50:
    for idx,img_path in enumerate(img_path_list):
        img_path = str(img_path)
        img = cv.imread(img_path)
        if img is not None:
            img_list.append(img)
            img_name = img_path.split('/')[-1]
            img_name_list.append(img_name)

    return img_list, img_name_list
def load_img_media(img_path_list):
    img_list, img_name_list = [], []
    # print(img_path_list[0])

    for idx,img_path in enumerate(img_path_list):
        img_path = str(img_path)
        img = cv.imread(img_path)
        if img is not None and (idx%5)==0:
            img_list.append(img)
            img_name = img_path.split('/')[-1]
            img_name_list.append(img_name)

    return img_list, img_name_list

# 生成中值图
def median_in_channel(img_list):
    shape = img_list[0].shape
    imgs = np.stack(img_list)
    # del img_list
    if len(shape)==3 and shape[-1] == 3:

        b = np.median(imgs[:, :, :, 0], axis=0)
        g = np.median(imgs[:, :, :, 1], axis=0)
        r = np.median(imgs[:, :, :, 2], axis=0)
        img = np.stack((b,g,r), axis=2).astype(int)
    else:
        img = np.median(imgs, axis=0).astype(int)
    return img


def get_fill_color(img):
    img = cv.resize(img,(512,512))
    line_0 = img[:,0,:]
    line_last = img[:,-1,:]
    col_0 = img[0,:,:]
    col_last = img[-1,:,:]

    stacked = np.stack((line_0,line_last,col_0,col_last),0)
    b = np.median(stacked[:,:,0])
    g = np.median(stacked[:,:,1])
    r = np.median(stacked[:,:,2])
    return b,g,r
# 膨胀alpha
def enlarge_line_for_ng_alpha(ng_alpha):
    gaussed = cv.GaussianBlur(ng_alpha,(5,5),2)
    # cv2.imshow("out",gaussed)
    gaussed[gaussed>10] = 255
    gaussed[gaussed<=10] = 0
    return ng_alpha
def resize_img(img,interp = cv2.INTER_LINEAR, sum_pixel = 512 * 512):
    # h = img.shape[0]
    # w = img.shape[1]
    # h_new = int(32 * np.floor(h * np.sqrt(300000/(w * h))/32))
    # w_new = int(32 * np.floor(w * np.sqrt(300000/(w * h))/32))

    h = img.shape[0]
    w = img.shape[1]

    ratio = np.round(h / w,5)

    x = sympy.symbols("x")  # 申明未知数"x"
    new_w = int(sympy.solve([x * x * ratio - sum_pixel], x)[1][0])
    new_h = int(ratio * new_w)
    img = cv2.resize(img,(new_w,new_h),interpolation=interp)
    return img
def get_new_img_size(h,w, ref=512,divide=8):

    """

    Args:
        h: height of origin image
        w: width of origin image
        sum_pixel: output pixel

    Returns:

    """
    if h > w:
        ratio = w / h
        edge = max(math.ceil(ref / divide * ratio / divide),4) * divide * divide
        return edge, ref
    ratio = h / w
    edge = math.ceil(max(ref / divide * ratio / divide,4 )* divide * divide)
    return ref, edge

