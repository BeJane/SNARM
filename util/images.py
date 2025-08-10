import json
import os
import glob
import pickle

import cv2
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np

from dataset.config import texture_source_dir
from util.data_util import augmentation, perlin_noise

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImagesDataset(Dataset):
    def __init__(self, root, mode='RGB', transforms=None):
        self.transforms = transforms
        self.mode = mode
        self.filenames = sorted([*glob.glob(os.path.join(root, '**', '*.jpg'), recursive=True),
                                 *glob.glob(os.path.join(root, '**', '*.png'), recursive=True)])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with Image.open(self.filenames[idx]) as img:
            # mode = L : convert img into gray
            img = img.convert(self.mode)

        if self.transforms:
            img = self.transforms(img)

        return img


class LabeledImagesDataset(Dataset):
    def __init__(self, root, mode='RGB', train_transforms=None, mask_transforms=None, label=None, ng_index=None,
                 feature_folder=None, aug_feature_folder=None, semi_label=False, label_num=None, args=None,
                 k_ratio=None):
        if train_transforms is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = train_transforms
        self.mask_transform = mask_transforms
        self.mode = mode
        self.label = label
        self.semi_label = semi_label
        self.root = root
        self.aug_feature_folder = aug_feature_folder

        self.filenames = sorted([*glob.glob(os.path.join(root, '*.jpg')),
                                 *glob.glob(os.path.join(root, '*.png'))])
        if semi_label:  # 模拟人工bbox标注的gt
            self.mask_filenames = sorted([*glob.glob(os.path.join(root, args.semi_label_folder, '*.npy'))])
        elif mask_transforms is not None and os.path.exists(os.path.join(root,'origin_gt')):
            self.mask_filenames = sorted([*glob.glob(os.path.join(root, 'origin_gt', '*.jpg')),
                                          *glob.glob(os.path.join(root, 'origin_gt', '*.png'))])
        else:
            self.mask_filenames = sorted([*glob.glob(os.path.join(root, 'binary', '*.jpg')),
                                          *glob.glob(os.path.join(root, 'binary', '*.png'))])
        if isinstance(feature_folder, str):
            self.feature_filenames = sorted([*glob.glob(os.path.join(root, feature_folder, '*.npy'))])
            assert len(self.filenames) == len(self.feature_filenames)
            self.feature_filenames = [self.feature_filenames]
        # print(feature_folder)

        elif isinstance(feature_folder, list):
            self.feature_filenames = [sorted([*glob.glob(os.path.join(root, folder, '*.npy'))]) for folder in
                                      feature_folder]

            assert len(self.filenames) == len(
                self.feature_filenames[0]), f"{len(self.filenames)},{len(self.feature_filenames[0])},{root}"
        else:
            self.feature_filenames = None
        self.label_num = label_num

        if ng_index is not None:
            self.filenames = [self.filenames[i] for i in ng_index]
            self.mask_filenames = [self.mask_filenames[i] for i in ng_index]
            self.feature_filenames = [[i[j] for j in ng_index] for i in self.feature_filenames]

        if k_ratio is not None:
            random_index = np.random.choice(len(self.filenames), round(len(self.filenames) * k_ratio))
            self.filenames = [self.filenames[i] for i in random_index]
            self.mask_filenames = [self.mask_filenames[i] for i in random_index]
            if self.feature_filenames is not None:            self.feature_filenames = [[i[j] for j in random_index] for
                                                                                        i in self.feature_filenames]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        with Image.open(self.filenames[idx]) as img:
            img = img.convert(self.mode)

        if self.transform:
            img = self.transform(img)
        if self.semi_label:
            mask = np.load(self.mask_filenames[idx])
        else:
            with Image.open(self.mask_filenames[idx]) as mask:
                mask = mask.convert('L')
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)
        if self.label_num is not None:
            if idx < self.label_num:
                mask[mask == 0] = -1  # 负样本
            else:
                mask = torch.zeros_like(mask)  # 无标签样本
            # print(self.root, idx,self.label_num,torch.unique(mask))

        item = {"image": img,
                "mask": mask,

                "label": self.label,
                "filename": self.filenames[idx]}
        # print(self.filenames[idx])
        if self.feature_filenames is not None:
            features = []
            h, w = 0, 0
            for p in self.feature_filenames:
                feature_path = p[idx]
                # print(feature_path)
                if self.aug_feature_folder is not None:
                    aug_feature_path = [*glob.glob(os.path.join(self.root, self.aug_feature_folder,
                                                                os.path.basename(feature_path).replace('.npy',
                                                                                                       '_*.npy')))]
                    aug_feature_path.append(feature_path)
                    assert len(aug_feature_path) == 8, feature_path
                    np.random.shuffle(aug_feature_path)
                    feature_path = aug_feature_path[0]
                # print(feature_path)
                feature = torch.tensor(np.load(feature_path))

                h = max(feature.shape[1], h)
                w = max(feature.shape[2], w)
                features.append(feature)

            item['feature'] = torch.cat(features).float()
            # print(len(features),item['feature'].shape)
        return item


class SampleDataset(LabeledImagesDataset):
    def __init__(self, root, mode='RGB', train_transforms=None, mask_transforms=None, label=None, ng_index=None,
                 feature_folder=None, aug_feature_folder=None, semi_label=False, label_num=None, args=None,
                 k_ratio=None, seed=None, split='train', num_sample=10):
        super().__init__(root, mode, train_transforms, mask_transforms, label, ng_index, feature_folder,
                         aug_feature_folder, semi_label,
                         label_num, args, k_ratio)

        if seed is not None:
            np.random.RandomState(seed).shuffle(self.filenames)
            np.random.RandomState(seed).shuffle(self.mask_filenames)
            shuffle_indices = np.arange(len(self.filenames))
            np.random.RandomState(seed).shuffle(shuffle_indices)
            if split == 'train':
                self.filenames = self.filenames[:num_sample]
                self.mask_filenames = self.mask_filenames[:num_sample]
                shuffle_indices = shuffle_indices[0:num_sample]
                # print(self.filenames)
            if split == 'test':
                self.filenames = self.filenames[num_sample:]
                self.mask_filenames = self.mask_filenames[num_sample:]
                shuffle_indices = shuffle_indices[num_sample:]
            if feature_folder is not None:
                self.feature_filenames = [[i[j] for j in shuffle_indices] for i in self.feature_filenames]


class SampleAugDataset(LabeledImagesDataset):
    def __init__(self, root, mode='RGB', train_transforms=None, mask_transforms=None, label=None, ng_index=None,
                 semi_label=False, label_num=None, args=None,rotate_90=True,
                 k_ratio=None, seed=None,  num_sample=10):
        super().__init__(root, mode, train_transforms, mask_transforms, label, ng_index, semi_label,
                         label_num, args, k_ratio)

        if seed is not None:
            np.random.RandomState(seed).shuffle(self.filenames)
            np.random.RandomState(seed).shuffle(self.mask_filenames)
            shuffle_indices = np.arange(len(self.filenames))
            np.random.RandomState(seed).shuffle(shuffle_indices)

        self.dtd_paths = sorted(glob.glob(texture_source_dir + "/*/*.jpg"))
        self.filenames = self.filenames[:num_sample]
        self.mask_filenames = self.mask_filenames[:num_sample]
        self.rotate_90 = rotate_90
        if self.label == 0:
            # Data augmentation
            self.augmented_data = []
            assert len(self.filenames) == num_sample
            for path in self.filenames:
                with Image.open(path) as img:
                    img = img.convert(self.mode)

                if self.transform:
                    img = self.transform(img)
                augmented_imgs = augmentation(img)
                for aug_img in augmented_imgs:
                    self.augmented_data.append(aug_img)

    def __len__(self):
        if self.label == 0:
            return len(self.augmented_data)
        if self.label == 1:
            return len(self.filenames)

    def __getitem__(self, idx):
        if self.label == 0:
            image = self.augmented_data[idx]

            mask = torch.zeros((1,image.shape[1],image.shape[2]))
        if self.label == 1:
            image = Image.open(self.filenames[idx]).convert("RGB")

            dtd_index = torch.randint(0, len(self.dtd_paths), (1,)).item()
            dtd_image = Image.open(self.dtd_paths[dtd_index]).convert("RGB")
            dtd_image = dtd_image.resize((image.height,image.width), Image.BILINEAR)
            fill_color = (114, 114, 114)
            # rotate_90
            if self.rotate_90:
                degree = np.random.choice(np.array([0, 90, 180, 270]))
                image = image.rotate(
                    degree, fillcolor=fill_color, resample=Image.BILINEAR
                )


            # perlin_noise implementation
            image, mask = perlin_noise(image, dtd_image, aug_prob=1.0)
            # matplotlib.use("Agg")
            # plt.subplot(1, 2, 1)
            # plt.imshow(image)
            # plt.subplot(1, 2, 2)
            # plt.imshow(mask[0])
            # plt.savefig(f"/media/szcycy/a/tmp/{idx}.png")
            image = self.transform(image)
            mask = transforms.ToTensor()(mask[0])
            # print(np.unique(mask))

        return {"image":image, "mask":mask,"label":self.label }


class PointDataset(Dataset):
    def __init__(self, root, mode='RGB', train_transforms=None, mask_transforms=None, label=None, ng_index=None,
                 feature_folder=None, aug_feature_folder=None, semi_label=False, label_num=None, args=None):
        if train_transforms is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = train_transforms
        self.mask_transform = mask_transforms
        self.mode = mode
        self.label = label
        self.semi_label = semi_label
        self.root = root
        self.aug_feature_folder = aug_feature_folder

        self.filenames = sorted([*glob.glob(os.path.join(root, '*.jpg')),
                                 *glob.glob(os.path.join(root, '*.png'))])
        if semi_label:  # selected points
            self.plabel_filenames = sorted([*glob.glob(os.path.join(root, args.semi_label_folder, '*.npy'))])
        if mask_transforms is not None:
            self.mask_filenames = sorted([*glob.glob(os.path.join(root, 'origin_gt', '*.jpg')),
                                          *glob.glob(os.path.join(root, 'origin_gt', '*.png'))])
        else:
            self.mask_filenames = sorted([*glob.glob(os.path.join(root, 'binary', '*.jpg')),
                                          *glob.glob(os.path.join(root, 'binary', '*.png'))])
        if isinstance(feature_folder, str):
            self.feature_filenames = sorted([*glob.glob(os.path.join(root, feature_folder, '*.npy'))])
            assert len(self.filenames) == len(self.feature_filenames)
            self.feature_filenames = [self.feature_filenames]
        # print(feature_folder)

        elif isinstance(feature_folder, list):
            self.feature_filenames = [sorted([*glob.glob(os.path.join(root, folder, '*.npy'))]) for folder in
                                      feature_folder]
            assert len(self.filenames) == len(
                self.feature_filenames[0]), f"{len(self.filenames)},{len(self.feature_filenames[0])}"
        else:
            self.feature_filenames = None
        self.label_num = label_num

        if ng_index is not None:
            self.filenames = [self.filenames[i] for i in ng_index]
            self.mask_filenames = [self.mask_filenames[i] for i in ng_index]
            self.feature_filenames = [[i[j] for j in ng_index] for i in self.feature_filenames]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with Image.open(self.filenames[idx]) as img:
            img = img.convert(self.mode)

        if self.transform:
            img = self.transform(img)

        with Image.open(self.mask_filenames[idx]) as mask:
            mask = mask.convert('L')
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)
        if self.label_num is not None:
            if idx < self.label_num:
                mask[mask == 0] = -1  # 负样本
            else:
                mask = torch.zeros_like(mask)  # 无标签样本
            # print(self.root, idx,self.label_num,torch.unique(mask))

        item = {"image": img,
                "mask": mask,

                "label": self.label,
                "filename": self.filenames[idx]}

        if self.feature_filenames is not None:
            features = []
            h, w = 0, 0
            for p in self.feature_filenames:

                feature_path = p[idx]
                # print(feature_path)
                if self.aug_feature_folder is not None:
                    aug_feature_path = [*glob.glob(os.path.join(self.root, self.aug_feature_folder,
                                                                os.path.basename(feature_path).replace('.npy',
                                                                                                       '_*.npy')))]
                    aug_feature_path.append(feature_path)
                    assert len(aug_feature_path) == 8, feature_path
                    np.random.shuffle(aug_feature_path)
                    feature_path = aug_feature_path[0]
                # print(feature_path)
                feature = torch.tensor(np.load(feature_path))
                h = max(feature.shape[1], h)
                w = max(feature.shape[2], w)
                features.append(feature)

            # for i in range(len(features)):
            #
            #     if features[i].shape[1] != h and features[i].shape[2] != w:
            #         features[i] = torch.nn.functional.interpolate(features[i].unsqueeze(1),
            #     size=(h,w),
            #     mode="bilinear",
            #     align_corners=False).squeeze(1)
            # print(features[i].shape,h,w)
            # if len(features) > 1:
            # features[i] = torch.nn.functional.normalize(features[i].permute(1,2,0), p=2.0, dim=2, eps=1e-12, out=None).permute(2,0,1)
            # print(torch.sqrt(torch.sum(features[i].permute(1,2,0)[0,0]**2)),features[i].permute(1,2,0)[0,0].shape)
            # print(len(features))
            item['feature'] = torch.cat(features).float()
            # print(len(features),item['feature'].shape)
        if self.semi_label:
            mask = np.load(self.plabel_filenames[idx])
            mask = transforms.ToTensor()(mask)
            item["p_label"] = mask
        else:
            item["p_label"] = mask  # torch.zeros((1,h,w))
        return item


class ErrorPointDataset(Dataset):
    def __init__(self, root, mode='RGB', train_transforms=None, mask_transforms=None, label=None, ng_index=None,
                 feature_folder=None, aug_feature_folder=None, semi_label=False, label_num=None, args=None):
        if train_transforms is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = train_transforms
        self.mask_transform = mask_transforms
        self.mode = mode
        self.label = label
        self.semi_label = semi_label
        self.root = root
        self.aug_feature_folder = aug_feature_folder

        self.filenames = sorted([*glob.glob(os.path.join(root, '*.jpg')),
                                 *glob.glob(os.path.join(root, '*.png'))])
        if semi_label:  # selected points
            self.plabel_filenames = sorted([*glob.glob(os.path.join(root, args.semi_label_folder, '*.npy'))])
        if mask_transforms is not None:
            self.mask_filenames = sorted([*glob.glob(os.path.join(root, 'origin_gt', '*.jpg')),
                                          *glob.glob(os.path.join(root, 'origin_gt', '*.png'))])
        else:
            self.mask_filenames = sorted([*glob.glob(os.path.join(root, 'binary', '*.jpg')),
                                          *glob.glob(os.path.join(root, 'binary', '*.png'))])
        if isinstance(feature_folder, str):
            self.feature_filenames = sorted([*glob.glob(os.path.join(root, feature_folder, '*.npy'))])
            assert len(self.filenames) == len(self.feature_filenames)
            self.feature_filenames = [self.feature_filenames]
        # print(feature_folder)

        elif isinstance(feature_folder, list):
            self.feature_filenames = [sorted([*glob.glob(os.path.join(root, folder, '*.npy'))]) for folder in
                                      feature_folder]
            assert len(self.filenames) == len(
                self.feature_filenames[0]), f"{len(self.filenames)},{len(self.feature_filenames[0])}"
        else:
            self.feature_filenames = None
        self.label_num = label_num

        if ng_index is not None:
            self.filenames = [self.filenames[i] for i in ng_index]
            self.mask_filenames = [self.mask_filenames[i] for i in ng_index]
            self.feature_filenames = [[i[j] for j in ng_index] for i in self.feature_filenames]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with Image.open(self.filenames[idx]) as img:
            img = img.convert(self.mode)

        if self.transform:
            img = self.transform(img)

        with Image.open(self.mask_filenames[idx]) as mask:
            mask = mask.convert('L')
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)
        if self.label_num is not None:
            if idx < self.label_num:
                mask[mask == 0] = -1  # 负样本
            else:
                mask = torch.zeros_like(mask)  # 无标签样本
            # print(self.root, idx,self.label_num,torch.unique(mask))

        item = {"image": img,
                "mask": mask,

                "label": self.label,
                "filename": self.filenames[idx]}

        if self.feature_filenames is not None:
            features = []
            h, w = 0, 0
            for p in self.feature_filenames:
                feature_path = p[idx]
                # print(feature_path)
                if self.aug_feature_folder is not None:
                    aug_feature_path = [*glob.glob(os.path.join(self.root, self.aug_feature_folder,
                                                                os.path.basename(feature_path).replace('.npy',
                                                                                                       '_*.npy')))]
                    aug_feature_path.append(feature_path)
                    assert len(aug_feature_path) == 8, feature_path
                    np.random.shuffle(aug_feature_path)
                    feature_path = aug_feature_path[0]
                # print(feature_path)
                feature = torch.tensor(np.load(feature_path))
                h = max(feature.shape[1], h)
                w = max(feature.shape[2], w)
                features.append(feature)

            item['feature'] = torch.cat(features).float()
            # print(len(features),item['feature'].shape)
        if self.semi_label:
            mask = np.load(self.plabel_filenames[idx])
            mask = torch.tensor(mask)
            item["p_label"] = mask
        else:
            item["p_label"] = -torch.ones((48, 3))
        return item


class ImagesFeaturesDataset(Dataset):
    def __init__(self, root, mode='RGB', transforms=None, label=None):
        self.transform = transforms
        self.mode = mode
        self.label = label
        self.filenames = sorted([*glob.glob(os.path.join(root, '*.jpg'), recursive=True),
                                 *glob.glob(os.path.join(root, '*.png'), recursive=True)])

        self.feature_filenames = sorted([*glob.glob(os.path.join(root, '*.npy'), recursive=True)])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with Image.open(self.filenames[idx]) as img:
            img = img.convert(self.mode)

        feature = np.load(self.feature_filenames[idx])
        if self.transform:
            img = self.transform(img)
            feature = torch.Tensor(feature)
        # return img,self.label
        return {"image": img,
                "feature": feature,
                # "label": self.label,
                "filename": self.filenames[idx]}


class FeatureDataset(Dataset):
    def __init__(self, root, mode='RGB', transforms=None):
        self.transforms = transforms
        self.mode = mode
        self.filenames = sorted([*glob.glob(os.path.join(root, '*.npz'))])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        item = np.load(self.filenames[idx])

        return item



class SurfaceTextDataset(Dataset):
    def __init__(self, root, data_info, mode='RGB', train_transforms=None, mask_transforms=None, label=None,
                 ng_index=None,
                 feature_folder=None, semi_label=False, label_num=None, args=None):
        if train_transforms is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = train_transforms
        self.mask_transform = mask_transforms
        self.mode = mode
        self.label = label
        self.semi_label = semi_label
        self.root = root
        self.info = data_info
        self.args = args

        self.filenames = sorted([*glob.glob(os.path.join(root, '*.jpg')),
                                 *glob.glob(os.path.join(root, '*.png'))])
        if semi_label:  # selected points
            self.plabel_filenames = sorted([*glob.glob(os.path.join(root, args.semi_label_folder, '*.npy'))])
        if mask_transforms is not None:
            self.mask_filenames = sorted([*glob.glob(os.path.join(root, 'origin_gt', '*.jpg')),
                                          *glob.glob(os.path.join(root, 'origin_gt', '*.png'))])
        else:
            self.mask_filenames = sorted([*glob.glob(os.path.join(root, 'binary', '*.jpg')),
                                          *glob.glob(os.path.join(root, 'binary', '*.png'))])
        if isinstance(feature_folder, str):
            self.feature_filenames = sorted([*glob.glob(os.path.join(root, feature_folder, '*.npy'))])
            assert len(self.filenames) == len(self.feature_filenames)
            self.feature_filenames = [self.feature_filenames]
        # print(feature_folder)

        elif isinstance(feature_folder, list):
            self.feature_filenames = [sorted([*glob.glob(os.path.join(root, folder, '*.npy'))]) for folder in
                                      feature_folder]
            assert len(self.filenames) == len(
                self.feature_filenames[0]), f"{len(self.filenames)},{len(self.feature_filenames[0])}"
        else:
            self.feature_filenames = None
        self.label_num = label_num

        if ng_index is not None:
            self.filenames = [self.filenames[i] for i in ng_index]
            self.mask_filenames = [self.mask_filenames[i] for i in ng_index]
            self.feature_filenames = [[i[j] for j in ng_index] for i in self.feature_filenames]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with Image.open(self.filenames[idx]) as img:
            img = img.convert(self.mode)

        if self.transform:
            img = self.transform(img)

        with Image.open(self.mask_filenames[idx]) as mask:
            mask = mask.convert('L')
        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = transforms.ToTensor()(mask)
        if self.label_num is not None:
            if idx < self.label_num:
                mask[mask == 0] = -1  # 负样本
            else:
                mask = torch.zeros_like(mask)  # 无标签样本
            # print(self.root, idx,self.label_num,torch.unique(mask))

        anomaly = '_'.join(os.path.basename(self.filenames[idx]).split('_')[3:-1])
        if 'false_ng' in self.filenames[idx]: anomaly = 'false_ng'
        if anomaly not in self.info['text'].keys():
            anomaly = 'ng'
        # print(anomaly)
        # if anomaly == 'good':
        #     tmp = []
        #     anomaly_keys = []
        #     for k in self.texts.keys():
        #         if k == 'good':continue
        #         anomaly_keys.append(k)
        #         # print(k)
        #         tmp.extend(self.texts[k][:40])
        #     np.random.shuffle(tmp)
        #     sens = tmp[:40]
        #
        #     kid = np.random.choice(len(anomaly_keys))
        #     avg_emb = self.avg_embs[f'{anomaly_keys[kid]}_embedding']
        #     avg_amask = self.avg_embs[f'{anomaly_keys[kid]}_attention_mask']
        # else:
        assert len(self.info['text'][anomaly]) >= 40, self.filenames[idx]
        sens = self.info['text'][anomaly][:40]
        avg_emb = self.info['avg_text'][f'{anomaly}_embedding']
        avg_amask = self.info['avg_text'][f'{anomaly}_attention_mask']

        item = {"image": img,
                "mask": mask,
                "label": self.label,
                "filename": self.filenames[idx],
                "sens": sens, "avg_emb": avg_emb.squeeze(0), "avg_amask": avg_amask.squeeze(0)}

        if self.feature_filenames is not None:
            features = []
            h, w = 0, 0
            for p in self.feature_filenames:
                feature_path = p[idx]
                # print(feature_path)

                feature = torch.tensor(np.load(feature_path))
                h = max(feature.shape[1], h)
                w = max(feature.shape[2], w)
                features.append(feature)

            item['feature'] = torch.cat(features).float()
            # print(len(features),item['feature'].shape)
        # get foreground

        origin_path = self.info['path'][os.path.basename(self.filenames[idx])]
        basename = os.path.basename(origin_path)
        dataset = origin_path.split('/')[0]
        fg_path = os.path.join(self.args.fg_dir, origin_path.replace(basename, f'f_{basename.replace("png", "npy")}'))
        if os.path.exists(fg_path):
            fg = np.load(fg_path)[
                 None, :, :]
            # print(query_fg.shape)
            knn_list = self.info['knn']['/'.join(origin_path.split('/')[1:])]
            if 'train' not in origin_path:
                for p in knn_list[:self.args.fg_knn]:
                    basename = os.path.basename(p)
                    ref_fg = np.load(
                        os.path.join(self.args.fg_dir, dataset,
                                     p.replace(basename, f'f_{basename.replace("png", "npy")}')))
                    fg = np.concatenate([fg, ref_fg[None, :, :]])
                # print(fg.shape)
            fg = np.max(fg, axis=0)
            fg = cv2.resize(fg, (h, w))
            fg[fg > 0.5] = 1
            fg[fg < 1] = 0
            fg = transforms.ToTensor()(fg)
        else:
            fg = torch.ones((1, h, w))
        # plt.imshow(fg[0,0])
        # plt.show()
        item['fg'] = fg
        if self.semi_label:
            mask = np.load(self.plabel_filenames[idx])
            mask = transforms.ToTensor()(mask)
            item["p_label"] = mask
        else:
            item["p_label"] = mask  # torch.zeros((1,h,w))
        return item
