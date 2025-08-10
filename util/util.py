import os.path
import random
import time
from pyexpat import features

import cv2
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Sampler
from tqdm import tqdm


def add_jitter(feature,scale=30):
    b,c,h,w = feature.shape
    feature_norms = (
            feature.norm(dim=1).unsqueeze(1) / c
    )
    jitter = torch.randn((b,c,h,w)).cuda()
    jitter = jitter * feature_norms * scale
    feature = feature + jitter
    return feature

def add_jitter_list(feature_list,scale=30):
    b, c, h, w = feature_list[0].shape
    jitter = torch.randn((b, c, h, w)).cuda()
    for i in range(len(feature_list)):

        feature_norms = (
                feature_list[i].norm(dim=1).unsqueeze(1) / c
        )

        jitter = jitter * feature_norms * scale
        feature_list[i] = feature_list[i] + jitter
    return feature_list

class RescaleSegmentor:
    def __init__(self, device, target_size=224,gaussian=True):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4
        self.gaussian = gaussian

    def convert_to_segmentation(self, patch_scores):

        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = torch.nn.functional.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()
        if not self.gaussian:
            return [patch_score for patch_score in patch_scores]
        # print(self.gaussian)
        return [
            ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
            for patch_score in patch_scores
        ]


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:

        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:

        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def predict_with_fg(loader,model, path_info, knn_info, args,l=None,use_image=False):
    outputsize = model.feature_size
    # print(outputsize)
    preds, gts, image_gts = [], [], []
    fgs = []
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch['image']
            features = batch['feature']
            masks = batch['mask']
            label = batch['label']
            filenames = batch['filename']
            masks[masks >= 0.5] = 1
            masks[masks < 1] = 0

            if use_image:
                pred = model(features,images)
            else:
                pred = model(features)
            #
            if args.num_classes == 1:
                pred = torch.sigmoid(pred)
            else:
                pred = torch.softmax(pred, dim=2)[:, :, 1] # 100,256,1
            if args.slide_window is not None:
                pred = pred.reshape(images.shape[0], -1, args.slide_window, args.slide_window)
                out = torch.zeros((images.shape[0], *outputsize),device='cuda')
                t = torch.zeros(outputsize,device='cuda')
                index = 0
                for i in range(0, outputsize[0] - args.slide_window + 1, args.slide_stride):
                    for j in range(0, outputsize[1] - args.slide_window + 1, args.slide_stride):
                        out[:, i:i + args.slide_window, j:j + args.slide_window] += pred[:, index]
                        t[i:i + args.slide_window, j:j + args.slide_window] += 1
                        index += 1
                pred = out / t
            # print(time.time()-start_time,features.shape)
            #     plt.subplot(1,3,1)
            #     plt.imshow(masks[0,0])
            #     plt.subplot(1,3,2)
            #     plt.imshow(out[0])
            #
            #     plt.subplot(1,3,3)
            #     plt.imshow(time)
            #     plt.show()
            if l is not None:
                features = features.numpy()
                features = np.sum(features,axis=1)
                # print(features.shape,pred.shape)
                pred = features*l+pred*(1-l)
            for i, filename in enumerate(filenames):
                origin_path = path_info[os.path.basename(filename)]
                basename = os.path.basename(origin_path)
                fg = np.load(os.path.join(args.fg_dir,origin_path.replace(basename,f'f_{basename.split(".")[0]+".npy"}')))[None,:,:]
                # print(query_fg.shape)
                knn_list = knn_info['/'.join(origin_path.split('/')[1:])]
                for p in knn_list[:args.fg_knn]:
                    basename = os.path.basename(p)
                    ref_fg = np.load(os.path.join(args.fg_dir,args.dataset,p.replace(basename,f'f_{basename.split(".")[0]+".npy"}')))
                    fg = np.concatenate([fg,ref_fg[None,:,:]])
                # print(fg.shape)
                fg = np.max(fg,axis=0)
                # plt.subplot(1,4,1)
                # plt.title('gt')
                # plt.imshow(masks[0,0])
                # plt.subplot(1,4,2)
                # plt.title('output')
                # plt.imshow(pred[0].cpu())
                # plt.subplot(1,4,3)
                # plt.title(f'knn fg(k={args.fg_knn})')
                # t = np.concatenate([pred.cpu().numpy(),cv2.resize(fg,(64,64))[None,:,:]])
                # t =np.min(t,axis=0)
                # plt.imshow(t)
                # plt.subplot(1,4,4)
                # plt.imshow(fg)
                # plt.show()
                fg = cv2.resize(fg,(masks.shape[-1],masks.shape[-2]))
                # fg = cv2.blur(fg,(20,20))
                # plt.imshow(fg)
                # plt.show()
                # pred[i] = pred[i] * torch.tensor(fg).cuda()
                fgs.append(torch.tensor(fg)[None,:,:])
            preds.append(pred)
            gts.append(masks)
            image_gts.append(label)
    preds = torch.cat(preds)  # 132 1024
    fgs = torch.cat(fgs)

    gts = torch.cat(gts).squeeze()  # 132,256,256
    image_gts = torch.cat(image_gts)
    anomaly_segmentor = RescaleSegmentor(
        device='cuda', target_size=gts.shape[-2:],gaussian=args.gaussian
    )

    preds = torch.reshape(preds, (-1, outputsize[0], outputsize[1]))
    preds = anomaly_segmentor.convert_to_segmentation(preds)
    preds = torch.tensor(np.array(preds))
    # print(fgs.shape,preds.shape)
    #
    preds = fgs**0.2 * preds
    # preds = (preds - torch.min(preds))/(torch.max(preds) - torch.min(preds))
    # plt.imshow(preds[0])
    # plt.show()
    preds = preds.cuda()
    image_scores = torch.max(torch.nn.functional.avg_pool2d(preds, 16, stride=2).view(preds.shape[0], -1), dim=-1)[0]

    # image_gts = np.max(masks, axis=(1, 2))
    return preds, gts, image_scores, image_gts

def predict(loader,model,args,l=None,use_image=False,use_prompt=False,use_cluster_feature=False):
    outputsize = model.feature_size
    # print(outputsize)
    preds, gts, image_gts = [], [], []
    image_scores =[]
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch['image']
            features = batch['feature']
            masks = batch['mask']
            label = batch['label']

            masks[masks >= 0.5] = 1
            masks[masks < 1] = 0
            if use_prompt:
                pred_list = []

                for prompt in batch['prompt']:
                    if use_image:
                        pred = model(features,images,prompt)
                    else:
                        pred = model(features,prompt)
                    pred = torch.softmax(pred, dim=2)[:, :, 1]
                    pred_list.append(pred)
                pred = torch.cat(pred_list)
                pred = torch.max(pred, dim=0)[0]
            else:

                if use_image:
                    pred = model(features, images)
                elif use_cluster_feature:
                    pred = model(features,batch['cluster_feature'])
                else:
                    pred = model(features)


                pred = torch.softmax(pred, dim=2)[:, :, 1] # 100,256,1
            if args.slide_window is not None:
                pred = pred.reshape(images.shape[0], -1, args.slide_window, args.slide_window)
                out = torch.zeros((images.shape[0], *outputsize),device='cuda')
                t = torch.zeros(outputsize,device='cuda')
                index = 0
                for i in range(0, outputsize[0] - args.slide_window + 1, args.slide_stride):
                    for j in range(0, outputsize[1] - args.slide_window + 1, args.slide_stride):
                        out[:, i:i + args.slide_window, j:j + args.slide_window] += pred[:, index]
                        t[i:i + args.slide_window, j:j + args.slide_window] += 1
                        index += 1
                pred = out / t
            # print(time.time()-start_time,features.shape)
            #     plt.subplot(1,3,1)
            #     plt.imshow(masks.cpu()[0,0])
            #     plt.subplot(1,3,2)
            #     plt.imshow(out.cpu()[0])
            #
            #     # plt.subplot(1,3,3)
            #     # plt.imshow(time)
            #     plt.show()
            if l is not None:
                features = features.numpy()
                features = np.sum(features,axis=1)
                # print(features.shape,pred.shape)
                pred = features*l+pred*(1-l)
            preds.append(pred)
            gts.append(masks)
            image_gts.append(label)
    preds = torch.cat(preds)  # 132 1024
    gts = torch.cat(gts).squeeze()  # 132,256,256
    image_gts = torch.cat(image_gts)
    anomaly_segmentor = RescaleSegmentor(
        device='cuda', target_size=gts.shape[-2:],gaussian=args.gaussian
    )
    preds = torch.reshape(preds, (-1, outputsize[0], outputsize[1]))


    preds = anomaly_segmentor.convert_to_segmentation(preds)
    preds = torch.tensor(np.array(preds)).cuda()
    image_scores = torch.max(torch.nn.functional.avg_pool2d(preds,16,stride=2).view(preds.shape[0], -1), dim=-1)[0]
    # image_gts = np.max(masks, axis=(1, 2))
    # plt.imshow(preds[0].cpu())
    # plt.show()
    # plt.title(args.dataset)
    # plt.hist(image_scores[image_gts==1].cpu().detach().numpy(), bins=10,label='ng')
    # plt.hist(image_scores[image_gts==0].cpu().numpy(), bins=10,label='ok')
    # plt.legend()
    # plt.show()
    # for i in range(preds.shape[0]):
        # if image_gts[i] == 1:
        # plt.subplot(1,3,1)
        # img = loader.dataset[i]['image'].numpy().transpose(1,2,0) * IMAGENET_STD + IMAGENET_MEAN
        # plt.imshow(img)
        # plt.subplot(1,3,2)
        # plt.imshow(gts[i],cmap='gray')
        # plt.subplot(1,3,3)
        # plt.imshow(preds[i].cpu())
        # plt.title(f'score: {image_scores[i]}')
        # plt.show()
    return preds, gts, image_scores, image_gts

class WeightEMA(object):
    """
    https://github.com/YU1ut/MixMatch-pytorch

    @article{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin},
  journal={arXiv preprint arXiv:1905.02249},
  year={2019}
}
    """
    def __init__(self, model, ema_model,lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset,idx_list,batch_size_list,steps_per_epoch=100):
        super(BalancedBatchSampler, self).__init__(dataset)

        self.steps_per_epoch = steps_per_epoch
        self.generator_list = []
        for idx in idx_list:
            self.generator_list.append(self.randomGenerator(idx))

        self.batch_size_list = batch_size_list

    def randomGenerator(self, list):
        while True:
            random_list = np.random.permutation(list)

            for i in random_list:
                yield i

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            batch = []
            for i,generator in enumerate(self.generator_list):
                # if i == 0:
                #     print(next(generator))
                for _ in range(self.batch_size_list[i]):
                    batch.append(next(generator))
            yield batch
def fix_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

