import math
import os
from glob import glob
import random
from typing import List

import cv2
import numpy as np
import torch
from einops import rearrange

import imgaug.augmenters as iaa
from matplotlib import pyplot as plt

from dataset.config import texture_source_dir

texture_source_file_list = sorted(glob(os.path.join(texture_source_dir,'*/*')))
def cut_paste_near(img, cluster, foreground_weight: np.ndarray = None,min_area=None):

    img = img.copy()
    h, w = img.shape[:2]
    min_area =h*w*0.01#0.005
    # plt.imshow(cluster)
    # plt.show()
    anomaly_img_mask = np.zeros((h, w), dtype=np.uint8)
    cluster = cv2.resize(cluster, (w, h), interpolation=cv2.INTER_NEAREST)
    perlin_mask = generate_perlin_noise_mask(h=img.shape[0], w=img.shape[1], perlin_noise_threshold=0.5)
    p_cn, p_cc_labels, _, _ = cv2.connectedComponentsWithStats((perlin_mask * 255).astype(np.uint8), connectivity=8)
    for p in range(1, p_cn):
        cur_anomaly_mask = p_cc_labels == p
        if np.sum(cur_anomaly_mask) < min_area:continue
        yp, xp = np.where(cur_anomaly_mask)
        # print(np.unique(cluster[yp,xp]),np.unique(cluster[yp,xp]).shape)
        # cluster_connectedComs = []
        # 针对每个cluster做一个
        p_cluster, p_cluster_count = np.unique(cluster[yp,xp],return_counts=True)
        # print(p_cluster,p_cluster_count,p_cluster[np.argmax(p_cluster_count)])
        choose_cluster = p_cluster[np.argmax(p_cluster_count)]
        cn1, cc_labels1, _, _ = cv2.connectedComponentsWithStats((cur_anomaly_mask*(cluster==choose_cluster) * 255).astype(np.uint8), connectivity=8)
        seg1 = cc_labels1 == 1
        if np.sum(seg1) < min_area:continue
        yp,xp = np.where(seg1)
        # 随机移动
        for t in range(10):
            offset_y, offset_x = random.randint(h // 32, h - 1) - (h // 10 + h - 1) * (
                    random.random() > 0.5), random.randint(w // 32, w - 1) - (w // 10 + w - 1) * (
                                         random.random() > 0.5)
            oy, ox = yp + offset_y, xp + offset_x
            oy, ox = oy % h, ox % w
            if choose_cluster not in np.unique(cluster[oy, ox]):
                continue
            # yp, xp = yp[(oy<h)*(ox<w)*(oy>=0)*(ox>=0)], xp[(oy<h)*(ox<w)*(oy>=0)*(ox>=0)]
            yp, xp = yp[cluster[oy, ox] != choose_cluster],xp[cluster[oy, ox] != choose_cluster]
            oy, ox = yp + offset_y, xp + offset_x

            oy, ox = oy % h, ox % w
            yp, xp = yp[foreground_weight[oy,ox] > 0.5],xp[foreground_weight[oy,ox] > 0.5]
            oy, ox = yp + offset_y, xp + offset_x
            oy, ox = oy % h, ox % w
            tmp = np.zeros_like(cluster)
            tmp[oy,ox] = 255
            cn_tmp, cc_labels_tmp, _, center_tmp = cv2.connectedComponentsWithStats(
                (tmp).astype(np.uint8), connectivity=8)
            if np.sum(cc_labels_tmp == 1) < min_area: continue
            yp, xp = yp[cc_labels_tmp[oy, ox] == 1], xp[cc_labels_tmp[oy, ox] ==1]
            oy, ox = yp + offset_y, xp + offset_x
            oy, ox = oy % h, ox % w
            # if np.unique(oy).shape[0] < 10 or np.unique(ox).shape[0]  < 10:continue
            # 添加异常
            # print(src_mask.shape)
            # print(np.round(center_tmp[1]))
            # plt.imshow(src_mask)
            # plt.show()
            # center = ((src_mask.shape[1]/2,src_mask.shape[0]//2))
            # src_mask = np.zeros_like(img)
            # src_mask[yp,xp] =[255,255,255]
            # src_mask = src_mask[np.min(yp):np.max(yp),np.min(xp):np.max(xp)]
            # center = (max(src_mask.shape[1]//2,min(w-src_mask.shape[1]//2,center_tmp[1][0].astype(int))),
            # max(src_mask.shape[0] // 2, min(h- src_mask.shape[0] // 2, center_tmp[1][1].astype(int))))
            # print(center,src_mask.shape)
            # img = cv2.seamlessClone(img[np.min(yp):np.max(yp),np.min(xp):np.max(xp)],img, src_mask,center,cv2.MIXED_CLONE )# cv2.NORMAL_CLONE)
            factor = random.uniform(0.8, 1)
            img[oy,ox] = img[yp,xp] * factor + (1 - factor) * img[oy,ox]
            anomaly_img_mask[oy,ox] = 1
            break
        # plt.subplot(1,2,1)
        # plt.imshow((cluster == cluster_connectedComs[near_j][2]).astype(int))
        # plt.subplot(1, 2, 2)
        # plt.imshow(img)
        # plt.show()
        if np.sum(anomaly_img_mask) > 250:
            return img.astype(np.uint8), anomaly_img_mask

    return img.astype(np.uint8), anomaly_img_mask
def cut_paste_near1(img, cluster, foreground_weight: np.ndarray = None):
    img = img.copy()
    h, w = img.shape[:2]
    # plt.imshow(cluster)
    # plt.show()
    anomaly_img_mask = np.zeros((h, w), dtype=np.uint8)
    cluster = cv2.resize(cluster, (w, h), interpolation=cv2.INTER_NEAREST)
    cluster_connectedComs = []
    # print(f'Cluster: {np.unique(cluster)}')
    # 针对每个cluster做一个
    for idx in np.unique(cluster):
        # perlin_mask = generate_perlin_noise_mask(h=img.shape[0],w=img.shape[1]).astype(bool)
        cur_cluster = (cluster == idx)
        # cur_mask = cur_cluster & (perlin_mask > 0)
        cn, cc_labels,_,_ = cv2.connectedComponentsWithStats((cur_cluster * 255).astype(np.uint8), connectivity=8)
        cluster_connectedComs.append((cn,cc_labels,idx))
    np.random.shuffle(cluster_connectedComs)

    for i in range(len(cluster_connectedComs)-1):
        for c in range(1,cluster_connectedComs[i][0]):
            y,x = np.where(cluster_connectedComs[i][1] == c)
            if y.shape[0] < h*w*0.1:continue
            if foreground_weight is not None:
                if (foreground_weight[y, x] > 0.5).mean() < 0.9:#保证前景
                    # if foreground_v < 0.9 or len(y) < (0.001 * h * w):
                    continue
            min_d = 999999999
            j_bg = False
            for j in range(i+1,len(cluster_connectedComs)):
                for cj in range(1,cluster_connectedComs[j][0]):
                    yj, xj = np.where(cluster_connectedComs[j][1] == cj)
                    if yj.shape[0] < h*w*0.1:continue
                    i_indexs = np.random.choice(y.shape[0],100,replace=False)
                    j_indexs = np.random.choice(yj.shape[0],100,replace=False)
                    for index in i_indexs:
                        d = np.min((y[index] - yj)**2 + (x[index] - xj)**2)
                        # print(((y[index] - yj)**2 + (x[index] - xj)**2).shape)
                        if d < min_d:
                            min_d = d
                            near_j = j
                            near_c = cj
                            if foreground_weight is not None and (foreground_weight[y, x] > 0.5).mean() < 0.9:
                                j_bg = True
            if min_d == 999999999:continue

            # anomaly_mask = np.zeros((h, w), dtype=np.uint8)
            perlin_mask = generate_perlin_noise_mask(h=img.shape[0], w=img.shape[1],perlin_noise_threshold=0.2)
            p_cn, p_cc_labels, _, _ = cv2.connectedComponentsWithStats((perlin_mask * 255).astype(np.uint8), connectivity=8)
            for p in range(1, p_cn):
                cur_anomaly_mask = p_cc_labels == p
                yp, xp = np.where(cur_anomaly_mask)
                # print(np.max(foreground_weight))
                # plt.imshow(foreground_weight)
                # plt.show()

                if c not in np.unique(cluster_connectedComs[i][1][yp,xp]) or \
                        (near_c not in np.unique(cluster_connectedComs[near_j][1][yp,xp]) and not j_bg):
                    continue

                    # plt.imshow(tmp_mask.astype(int)+cur_anomaly_mask)
                    # plt.show()
                    # anomaly_mask[cluster==cluster_connectedComs[near_j][3]] = 0
                    # anomaly_mask[cluster==i] = 1
                    # tmp_mask = cluster_connectedComs[near_j][1][yp,xp] == near_c
                    # oy = yp [tmp_mask ]
                    # ox = xp[ tmp_mask]
                cur_anomaly_mask[cluster==cluster_connectedComs[near_j][2]] = 0

                yp, xp = np.where(cur_anomaly_mask)
                if foreground_weight is not None:
                    foreground_v = (foreground_weight[yp, xp] > 0.5).mean()
                    if foreground_v < 0.9 or len(y) < (0.001 * h * w):
                        continue

                print(cluster_connectedComs[near_j][2],np.unique(cluster[yp,xp]))
                # 随机移动
                for t in range(10):
                    offset_y, offset_x = random.randint(h // 32, h - 1) - (h // 10 + h - 1) * (
                            random.random() > 0.5), random.randint(w // 32, w - 1) - (w // 10 + w - 1) * (
                                                 random.random() > 0.5)
                    oy, ox = yp + offset_y, xp + offset_x
                    oy, ox = oy % h, ox % w
                    if cluster_connectedComs[near_j][2] not in np.unique(cluster[oy, ox]):
                        continue
                    yp,xp = yp[cluster[oy, ox] == cluster_connectedComs[near_j][2] ],xp[cluster[oy, ox] == cluster_connectedComs[near_j][2] ]
                    oy, ox = yp + offset_y, xp + offset_x
                    oy, ox = oy % h, ox % w
                    # 添加异常

                    factor = random.uniform(0.8, 1)
                    img[yp, xp] =  img[oy,ox]* factor + (1 - factor) * img[yp, xp]
                    anomaly_img_mask[yp, xp] = 1
                    break
                # plt.subplot(1,2,1)
                # plt.imshow((cluster == cluster_connectedComs[near_j][2]).astype(int))
                # plt.subplot(1, 2, 2)
                # plt.imshow(img)
                # plt.show()
                if np.sum(anomaly_img_mask) > 250:
                    return img.astype(np.uint8), anomaly_img_mask
    # assert np.sum(anomaly_img_mask) > 10,np.sum(anomaly_img_mask)
    return img.astype(np.uint8), anomaly_img_mask
def cut_paste(img, cluster, foreground_weight: np.ndarray = None):
    img = img.copy()
    h, w = img.shape[:2]
    anomaly_img_mask = np.zeros((h, w), dtype=np.uint8)
    cluster = cv2.resize(cluster, (w, h), interpolation=cv2.INTER_NEAREST)
    # 针对每个cluster做一个
    for idx in np.unique(cluster):
        perlin_mask = generate_perlin_noise_mask(h=img.shape[0],w=img.shape[1]).astype(bool)
        cur_cluster = (cluster == idx)
        cur_mask = cur_cluster & (perlin_mask > 0)
        cn, cc_labels, _, _ = cv2.connectedComponentsWithStats((cur_mask * 255).astype(np.uint8), connectivity=8)
        for i in range(1, cn):
            cur_anomaly_mask = cc_labels == i
            y, x = np.where(cur_anomaly_mask)
            if foreground_weight is not None:
                foreground_v = (foreground_weight[y, x] > 0.5).mean()
                if foreground_v < 0.9 or len(y) < (0.001 * h*w):
                    cur_mask[cur_anomaly_mask] = 0
                    continue
            # 随机移动
            for i in range(10):
                offset_y, offset_x = random.randint(h // 32, h - 1) - (h // 5 + h - 1) * (
                            random.random() > 0.5), random.randint(w // 32, w - 1) - (w // 5 + w - 1) * (
                                                 random.random() > 0.5)
                oy, ox = y + offset_y, x + offset_x
                oy, ox = oy % h, ox % w
                if cur_cluster[oy, ox].mean() > 0.5:  # 还在当前cluster中
                    continue
                # 添加异常
                factor = random.uniform(0.5, 1)
                img[y, x] = img[oy, ox] * factor + (1 - factor) * img[y, x]
                break
            else:
                cur_mask[cur_anomaly_mask] = 0
        anomaly_img_mask[cur_mask] = 1

    return img.astype(np.uint8), anomaly_img_mask
def generate_anomaly(img: np.ndarray,target_foreground_mask=None,transparency_range = [0.3, 1]) -> List[np.ndarray]:
    '''
    MemSeg: A semi-supervised method for image surface defect detection using differences and commonalities
    step 1. generate mask
        - target foreground mask
        - perlin noise mask

    step 2. generate texture or structure anomaly
        - texture: load DTD
        - structure: we first perform random adjustment of mirror symmetry, rotation, brightness, saturation,
        and hue on the input image  𝐼 . Then the preliminary processed image is uniformly divided into a 4×8 grid
        and randomly arranged to obtain the disordered image  𝐼

    step 3. blending image and anomaly source
    '''
    h,w,_ = img.shape

    # step 1. generate mask

    ## target foreground mask
    if target_foreground_mask is None:
        target_foreground_mask = generate_target_foreground_mask(img=img)


    ## perlin noise mask
    perlin_noise_mask = generate_perlin_noise_mask(h=img.shape[0],w=img.shape[1])

    ## mask
    mask = perlin_noise_mask * target_foreground_mask
    mask_expanded = np.expand_dims(mask, axis=2)

    # step 2. generate texture or structure anomaly
    if np.min(target_foreground_mask) == 0: # 如果有前景信息
        p = np.random.uniform()
    else:
        p = 0
    p = 0
    if p < 0.8:
        ## anomaly source
        anomaly_source_img = anomaly_source(img=img)

        ## mask anomaly parts
        factor = np.random.uniform(*transparency_range, size=1)[0]
        anomaly_source_img = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * img)

        # step 3. blending image and anomaly source
        anomaly_source_img = ((- mask_expanded + 1) * img) + anomaly_source_img
    else:
        factor = 1
        while True:
            perlin_noise_mask = generate_perlin_noise_mask(h=img.shape[0], w=img.shape[1],perlin_noise_threshold=0.4)
            ## mask
            mask = perlin_noise_mask * target_foreground_mask
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(mask * 255), connectivity=8)
            if num_labels == 1:continue
            if np.array(stats)[1, -1] >= np.sum(target_foreground_mask)/16:
                break

        mask[labels == 1] = 1
        mask[labels != 1] = 0

        if p >= 0.75:
            height, width = img.shape[:2]  # 输入(H,W,C)，取 H，W 的值
            center = (width / 2, height / 2)  # 绕图片中心进行旋转
            angle = np.random.uniform(5,25)  # 旋转方向取（-180，180）中的随机整数值，负为逆时针，正为顺势针
            scale = 1  # 将图像缩放为80%

            # 获得旋转矩阵
            M = cv2.getRotationMatrix2D(center, angle, scale)

            # 进行仿射变换，边界填充为255，即白色，默认为0，即黑色
            bg_value = img[target_foreground_mask==0][0].tolist()
            anomaly_source_img = cv2.warpAffine(src=img, M=M, dsize=(height, width), borderValue=bg_value)
        else:

            anomaly_source_img = cv2.resize(img[target_foreground_mask == 0].reshape(-1, 1, 3),
                                            (img.shape[1], img.shape[0]))
        mask_expanded = np.expand_dims(mask, axis=2)

        anomaly_source_img = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * img)
        # step 3. blending image and anomaly source
        anomaly_source_img = ((- mask_expanded + 1) * img) + anomaly_source_img


    return (anomaly_source_img.astype(np.uint8), mask)

def anomaly_source(img: np.ndarray) -> np.ndarray:
    p = np.random.uniform()
    if p < 0.5:
        # TODO: None texture_source_file_list
        anomaly_source_img = _texture_source(h=img.shape[0],w=img.shape[1])
    else:
        anomaly_source_img = _structure_source(h=img.shape[0],w=img.shape[1],img=img)

    return anomaly_source_img

def _texture_source(h,w) -> np.ndarray:
    idx = np.random.choice(len(texture_source_file_list))
    texture_source_img = cv2.imread(texture_source_file_list[idx])
    texture_source_img = cv2.cvtColor(texture_source_img, cv2.COLOR_BGR2RGB)
    texture_source_img = cv2.resize(texture_source_img, dsize=(w,h)).astype(np.float32)

    return texture_source_img

def _structure_source(h,w, img: np.ndarray,structure_grid_size=8) -> np.ndarray:
    structure_source_img = rand_augment()(image=img)
    while h % structure_grid_size !=0:
        structure_grid_size += 2
    # assert h % structure_grid_size == 0, 'structure should be devided by grid size accurately'
    grid_w = w // structure_grid_size
    grid_h = h // structure_grid_size

    structure_source_img = rearrange(
        tensor=structure_source_img,
        pattern='(h gh) (w gw) c -> (h w) gw gh c',
        gw=grid_w,
        gh=grid_h
    )
    disordered_idx = np.arange(structure_source_img.shape[0])
    np.random.shuffle(disordered_idx)

    structure_source_img = rearrange(
        tensor=structure_source_img[disordered_idx],
        pattern='(h w) gw gh c -> (h gh) (w gw) c',
        h=structure_grid_size,
        w=structure_grid_size
    ).astype(np.float32)

    return structure_source_img

def rand_augment():
    augmenters = [
        iaa.GammaContrast((0.5, 2.0), per_channel=True),
        iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
        iaa.pillike.EnhanceSharpness(),
        iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
        iaa.Solarize(0.5, threshold=(32, 128)),
        iaa.Posterize(),
        iaa.Invert(),
        iaa.pillike.Autocontrast(),
        iaa.pillike.Equalize(),
        iaa.Affine(rotate=(-45, 45))
    ]

    aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
    aug = iaa.Sequential([
        augmenters[aug_idx[0]],
        augmenters[aug_idx[1]],
        augmenters[aug_idx[2]]
    ])

    return aug

def generate_target_foreground_mask(img: np.ndarray) -> np.ndarray:
    # convert RGB into GRAY scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # generate binary mask of gray scale image
    _, target_background_mask = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    target_background_mask = target_background_mask.astype(np.bool).astype(np.int)

    # invert mask for foreground mask
    target_foreground_mask = -(target_background_mask - 1)

    return target_foreground_mask

def generate_perlin_noise_mask(h,w,  perlin_scale=6,min_perlin_scale=0,perlin_noise_threshold=0.5) -> np.ndarray:
    # define perlin noise scale
    perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
    perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

    # generate perlin noise
    perlin_noise = rand_perlin_2d_np((h,w), (perlin_scalex, perlin_scaley))

    # apply affine transform
    rot = iaa.Affine(rotate=(-90, 90))
    perlin_noise = rot(image=perlin_noise)

    # make a mask by applying threshold
    mask_noise = np.where(
        perlin_noise > perlin_noise_threshold,
        np.ones_like(perlin_noise),
        np.zeros_like(perlin_noise)
    )

    return mask_noise
def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (math.ceil(shape[0] / res[0]), math.ceil(shape[1] / res[1]))
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients,d[0],axis=0),d[1],axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],d[0],axis=0),d[1],axis=1)
    dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)
    # print(d,gradients.shape,tile_grads([0, -1], [0, -1]).shape)
    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])

def lerp_np(x,y,w):
    fin_out = (y-x)*w + x
    return fin_out
