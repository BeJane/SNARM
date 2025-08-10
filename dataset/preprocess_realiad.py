import sys

import cv2

import os
import numpy as np

from tqdm import tqdm


import json

from util.util import fix_seed

sys.path.append('../')

from dataset.generate_anomaly import generate_anomaly

SEED = 0

resize = (448,448)
crop_size = (392,392)#w,h
root_dir = '/media/szcyxy/新加卷/qi_data/RealIAD(CVPR24)/realiad_1024'
exp_path = '/media/szcyxy/新加卷/qi_data/RealIAD(CVPR24)/realiad_jsons/realiad_jsons'

for single_dataset in sorted(os.listdir(root_dir)):
    if '.' in  single_dataset:continue
    fix_seed(SEED)

    # if   single_dataset not  in  path_info.keys():continue
    out_dir = f'../data/defect_392/realiad/{single_dataset}'
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/input_size.txt', 'w') as f:
        f.write(f'{crop_size[1]},{crop_size[0]}')
    dataset_info_path = f'{exp_path}/{single_dataset}.json'
    if not os.path.exists(dataset_info_path):continue

    print(single_dataset)
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)

    origin_paths = {}


    for dset in ['train','test']:
        ok_path_list,ok_mask_list,ng_path_list,ng_mask_list=[],[],[],[]
        for item in dataset_info[dset]:
            if item['anomaly_class'] == dataset_info["meta"]["normal_class"]:
                ok_path_list.append(item['image_path'])
            else:
                ng_path_list.append(item['image_path'])
                ng_mask_list.append(item['mask_path'])
        for category in ['false_ng','ng','ok']:
            if dset == 'test' and category == 'false_ng':continue
            os.makedirs(os.path.join(out_dir, dset,category,'binary'),exist_ok=True)
            if category != 'false_ng':
                os.makedirs(os.path.join(out_dir, dset,category,'origin_gt'),exist_ok=True)

            if dset == 'train' and category == 'false_ng':

                path_list =ok_path_list
                alpha_list = ok_mask_list
                
            elif category == 'ok':
                path_list = ok_path_list
                alpha_list = ok_mask_list
            elif category == 'ng':
                path_list = ng_path_list
                alpha_list = ng_mask_list

            image_id = 0
            for i, image_path in tqdm(enumerate(path_list)):
                info = image_path.split('/')[-2].replace('ko','surface')
                info = info.replace('ok','good')
                image_name = info + '_' +os.path.basename(image_path).split('.', 1)[0]
                # ref_path = ref_list[i]

                origin_img = cv2.imread(os.path.join(root_dir,single_dataset,image_path))
                # print(root_dir,image_path,os.path.join('root_dir',image_path))
                if i < len(alpha_list):
                    pha_path = alpha_list[i]
                    origin_pha= cv2.imread(os.path.join(root_dir,single_dataset,pha_path))
                    origin_pha[origin_pha>0]=255
                    # print(np.unique(origin_pha))
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
                    # fg_path = os.path.join(fg_dir, single_dataset, os.path.basename(image_path).split('.')[0]+ '.npy')
                    #
                    # fg = np.load(fg_path)
                    # fg = cv2.resize(fg, (input_size[0], input_size[-1]), interpolation=cv2.INTER_NEAREST)
                    # fg[fg > 0.5] = 1
                    # fg[fg <= 0.5] = 0
                    while True:
                        crop_image_, m = generate_anomaly(img=crop_image, target_foreground_mask=np.ones_like(crop_pha_image))
                        crop_pha_image = np.uint8(m * 255)
                        if np.max(crop_pha_image) < 255: continue
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(crop_pha_image,
                                                                                                connectivity=8, )
                        if np.min(np.array(stats)[:, -1]) > 0:  # input_size[0]*input_size[1]*0.005:

                            crop_image = crop_image_
                            break
                crop_pha_image[crop_pha_image >= 127] = 255
                crop_pha_image[crop_pha_image < 127] = 0


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