
import cv2

import os
import numpy as np
from tqdm import tqdm

import json
from dataset.generate_anomaly import generate_anomaly
from dataset.utils import get_new_img_size
from util.util import fix_seed

SEED = 0

resize = (448,448)
crop_size = (392,392)#w,h
root_dir = '/home/hanxi/qi_data/mvtec_3d'
out_root_dir = '../data/defect_392/mvtec3d'
# fg_dir='../work_dirs/coreset_boundary_fg_original'
exp_path='mvtec3d_index.json'
with open(exp_path,'r',encoding='utf-8') as f:
    path_info = json.load(f)

for single_dataset in sorted(os.listdir(root_dir)):
    fix_seed(SEED)
# for single_dataset in ['BS','KolektorSDD','KolektorSDD2','magnetic_tile','cam0','cam2','cam3','LAO','YITENG']:
    # if single_dataset in ['bottle','leather','capsule']:continue
    if '.' in  single_dataset:continue
    out_dir = f'{out_root_dir}/{single_dataset}'
    os.makedirs(out_dir,exist_ok=True)


    print(single_dataset)
    with open(f'{out_dir}/input_size.txt', 'w') as f:
        f.write(f'{crop_size[1]},{crop_size[0]}')

    origin_paths = {}
    # ng_path_list = path_info[single_dataset]['test']['ng'] #+ path_info[single_dataset]['train']['ng']
    # ng_alpha_list = path_info[single_dataset]['test']['ng_binary'] #+ path_info[single_dataset]['train']['ng_binary']
    # np.random.RandomState(SEED).shuffle(ng_path_list)
    # np.random.RandomState(SEED).shuffle(ng_alpha_list)


    for dset in ['train','test']:
        for category in ['ok','ng','false_ng']:
            if dset == 'test' and category == 'false_ng':continue
            os.makedirs(os.path.join(out_dir, dset,category,'binary'),exist_ok=True)
            if category != 'false_ng':
                os.makedirs(os.path.join(out_dir, dset,category,'origin_gt'),exist_ok=True)

            if dset == 'train' and category == 'false_ng':
                path_list = path_info[single_dataset][dset]['ok']*2
                alpha_list = path_info[single_dataset][dset][f'ok_binary']*2

            else:
                path_list = path_info[single_dataset][dset][category]
                alpha_list = path_info[single_dataset][dset][f'{category}_binary']
            fg_list = []  # path_info[single_dataset][dset][f'{category}_fg']
            image_id = 0

            for i, image_path in tqdm(enumerate(path_list)):
                image_name = image_path.split('/')[-3] + '_' +os.path.basename(image_path).split('.', 1)[0]
                # ref_path = ref_list[i]

                origin_img = cv2.imread(os.path.join(root_dir,image_path))
                # print(root_dir,image_path,os.path.join('root_dir',image_path))
                if i < len(alpha_list):
                    pha_path = alpha_list[i]
                    origin_pha= cv2.imread(os.path.join(root_dir,pha_path))
                else:
                    origin_pha = np.zeros_like(origin_img)



                H,W = origin_img.shape[:2]
                # origin_ref = cv2.resize(origin_ref,(W,H),cv2.INTER_NEAREST)
                image_name = f'{dset}_{category}_{image_id}_' + image_name
                image_id += 1
                center = (resize[0] // 2, resize[1] // 2)

                crop_image = cv2.resize(origin_img, (resize[0], resize[-1]))[
                             center[0] - crop_size[0] // 2:center[0] + crop_size[0] // 2,
                             center[1] - crop_size[1] // 2:center[1] + crop_size[1] // 2]

                crop_pha_image = cv2.resize(origin_pha, (resize[0], resize[-1]))[
                                 center[0] - crop_size[0] // 2:center[0] + crop_size[0] // 2,
                                 center[1] - crop_size[1] // 2:center[1] + crop_size[1] // 2]

                crop_pha_image = cv2.cvtColor(crop_pha_image, cv2.COLOR_BGR2GRAY)
                if dset == 'train' and category == 'false_ng':
                    # fg_path = os.path.join(fg_dir, single_dataset, image_name.split('_')[-1] + '.npy')
                    #
                    # fg = np.load(fg_path)
                    # fg  = cv2.resize(fg,(input_size[0],input_size[-1]),interpolation=cv2.INTER_NEAREST)
                    # fg[fg >0.5]=1
                    # fg[fg <= 0.5]=0
                    while True:
                        crop_image_,m = generate_anomaly(img=crop_image,target_foreground_mask=np.ones_like(crop_pha_image))
                        crop_pha_image = np.uint8(m * 255)
                        if np.max(crop_pha_image) < 255:continue
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(crop_pha_image, connectivity=8,)
                        if  np.min(np.array(stats)[:,-1]) > 0:#input_size[0]*input_size[1]*0.005:

                            crop_image =crop_image_
                            break
                # ymin,ymax,xmin,xmax = 16,240,16,240
                # crop_image = crop_image[ymin:ymax, xmin:xmax]  # 裁剪图片

                # crop_pha_image = crop_pha_image[ymin:ymax, xmin:xmax]
                # crop_fg_image = crop_fg_image[ymin:ymax, xmin:xmax]

                crop_pha_image[crop_pha_image >= 127] = 255  # 去除噪声干扰
                crop_pha_image[crop_pha_image < 127] = 0

                # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(crop_pha_image, connectivity=8)
                # if dset == 'test' and category == 'ng' and np.min(np.array(stats)[:,-1]) < input_size[0]*input_size[1]*0.005 :
                #     print(image_path)
                #     continue

                save_image_name = f'{image_name}.png'

                origin_paths[save_image_name] = image_path
                # save_median_image_name = f'{image_name}_median.png'
                # save_image_name = filter_file_name(save_image_name)
                cv2.imwrite(os.path.join(
                    out_dir,dset,category, save_image_name), crop_image)
                # cv2.imwrite(os.path.join(
                #     out_dir, dset,save_median_image_name), crop_median_image)
                save_pha_image_name = f'{image_name}_pha.png'
                if category != 'false_ng':
                    cv2.imwrite(os.path.join(
                        out_dir, dset, category, 'origin_gt', save_pha_image_name), origin_pha)
                # else:
                cv2.imwrite(os.path.join(
                    out_dir,  dset,category,'binary',save_pha_image_name), crop_pha_image)


    with open(os.path.join(out_dir,'origin_path.json'), 'w', encoding="utf-8") as f:
        json.dump(origin_paths, f, indent=4, ensure_ascii=False)