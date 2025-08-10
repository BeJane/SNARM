import logging
import math
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from . import  common

from .pca_torch import PCA_torch
from .pos_embed import get_2d_sincos_pos_embed

LOGGER = logging.getLogger(__name__)


class InterMatcher(torch.nn.Module):
    def __init__(self, device, save_patch_scores=False,save_residuals=False,ref_num_patches=None,add_pos_embed=True,
           visible=False,pos_weight=1,mode='cpu',pca_com=0.95,num_groups=2):
        super(InterMatcher, self).__init__()
        self.device = device
        self.save_patch_scores = save_patch_scores
        self.save_residuals = save_residuals
        self.ref_num_patches = ref_num_patches

        self.pos_embed = None
        self.add_pos_embed = add_pos_embed

        self.pos_weight = pos_weight
        self.pca_com = pca_com
        self.mode = mode
        self.visible = visible
        self.num_groups = num_groups
        print(f'Add position embedding: {add_pos_embed}! Weight={self.pos_weight} ! Num groups={self.num_groups}')


    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=None,
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape

        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator
        # if self.mode == 'gpu':

        self.anomaly_scorer = common.NearestNeighbourScorer_gpu(
            n_nearest_neighbours=anomaly_score_num_nn
        )

        # else:
        #     self.anomaly_scorer = common.NearestNeighbourScorer(
        #     n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        # )

        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images,context=None,provide_patch_shapes=False,mode='gpu'):
        """Returns feature embeddings for images."""

        def fuse_feature( feat_list):
            return torch.stack(feat_list, dim=1).mean(dim=1)
        def _detach(features):
            # if detach:
            if mode == 'gpu':
                return features.detach()
            return features.cpu().detach()
            # return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)
        # print(features[self.layers_to_extract_from[0]].shape) # b,c,h,w
        features = [features[layer] for layer in self.layers_to_extract_from]
        if features[0].dim() == 3:
            group_size = len(features)//self.num_groups
            features = [fuse_feature(features[g*group_size:(g+1)*group_size]) for g in range(self.num_groups)]
            # if self.num_groups == 1:
            #     features = [fuse_feature([features[i] for i in range(group_size)])]
            features = [f.transpose(1,2)[:,:,1+self.backbone.num_register_tokens:] for f in features]
            features = [f.reshape(*f.shape[:2],int(math.sqrt(f.shape[2])),int(math.sqrt(f.shape[2]))) for f in features]
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        # print(patch_shapes)# [[32, 32], [16, 16]]
        features = [x[0] for x in features]
        if self.ref_num_patches is None:
            self.ref_num_patches = patch_shapes[0]
        for i in range(0, len(features)):

            _features = features[i]
            patch_dims = patch_shapes[i]
            if (patch_dims[0],patch_dims[1]) == self.ref_num_patches :continue
            # TODO(pgehler): Add comments
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape

            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(self.ref_num_patches[0], self.ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], self.ref_num_patches[0], self.ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features # b,1024,1024,3,3
            # print(features[i].shape)
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # print(features[0].shape) # 8192,512,3,3
        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)# 8192,1024
        if self.add_pos_embed:
            if self.pos_embed is None:
                self.pos_embed = get_2d_sincos_pos_embed(features.shape[-1],self.ref_num_patches,cls_token=False)
                self.pos_embed = torch.tensor(self.pos_embed)
            features = features.reshape(-1,self.pos_embed.shape[0],features.shape[-1]) + self.pos_embed.type_as(features)*self.pos_weight
            features =features.reshape(-1,features.shape[-1])


        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data,bankpath):
        """PatchCore training.

        This function computes the embeddings of the training data and fills the
        memory bank of SPADE.
        """
        self._fill_memory_bank(training_data,bankpath)

    def _fill_memory_bank(self, input_data,bankpath):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image,context=None):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image,context,mode=self.mode)
        if bankpath is not  None and os.path.exists(bankpath):
            print(f"Loading bank from {bankpath}")
            with open(bankpath,'rb') as f:
                info = pickle.load(f)
                self.sample_indices = info['sample_indices']
                self.sample_features = info['sample_features']
                self.sample_pca_features = info['sample_pca_features']
                self.sample_image_indices = info['sample_image_indices']
                self.pca = info['pca']
                self.images = info['images']
        else:
            features = []
            images = []


            with tqdm.tqdm(
                input_data, desc="Computing support features...", position=1, leave=False
            ) as data_iterator:
                for batch in data_iterator:
                    if isinstance(batch, dict):
                        image = batch["image"]

                    images.append(image)
                    patch_features = _image_to_features(image)
                    for  patch in patch_features:
                        features.append(patch.unsqueeze(0))
            if len(features) > 1000000:
                subset_indices = np.random.choice(
                    len(features), 1000000, replace=False
                )
                features=[features[i] for i in subset_indices]
                if self.mode == 'gpu':
                    features = torch.concat(features, dim=0)
                if self.mode == 'cpu':
                    features = np.concatenate(features, axis=0)
                    features = torch.from_numpy(features)
                features, sample_indices = self.featuresampler.run(features.cuda())
                sample_indices = subset_indices[sample_indices]
            else:
                if self.mode == 'gpu':
                    features = torch.concat(features, dim=0)
                if self.mode == 'cpu':
                    features = np.concatenate(features, axis=0)
                    features = torch.from_numpy(features)
                features, sample_indices = self.featuresampler.run(features.cuda())
            self.pca = PCA_torch(n_components=self.pca_com)
            newX = self.pca.fit_transform(features.cpu().numpy())
            print(f'PCA: {newX.shape}')

            # self.anomaly_scorer.fit(detection_features=[features],pca_features=[torch.from_numpy(newX).type_as(features)])
            self.sample_indices = sample_indices
            self.sample_features = features
            self.sample_pca_features = torch.tensor(newX).type_as(features)
            self.sample_image_indices = self.sample_indices // (self.ref_num_patches[0] * self.ref_num_patches[1])
            self.sample_image_indices = torch.tensor(self.sample_image_indices).type_as(features)

            images = np.concatenate(images, axis=0)  # (213, 3, 512, 512)
            stride =images.shape[2]//self.ref_num_patches[0]
            images = torch.nn.functional.unfold(torch.tensor(images), stride,
                                                stride=stride).transpose(1, 2)
            images = images.reshape(-1, 3,stride,stride)
            self.images = images[sample_indices]
            if bankpath is not None:
                with open(bankpath,'wb') as f:
                    pickle.dump({'sample_indices':sample_indices,
                                 'sample_features':features,
                                 'sample_pca_features':self.sample_pca_features,
                                 'sample_image_indices': self.sample_image_indices,
                                 'pca':self.pca,
                                 'images':self.images},f)

    def _predict(self, images,ids=None, topk=1,with_image_features=False):

        # indices = np.where(np.isin(self.sample_image_indices,ids))[0]
        if ids is not None:
            indices = torch.where(torch.isin(self.sample_image_indices.cuda(),torch.Tensor(ids).cuda()))[0]
            ref_features = self.sample_features.cuda()[indices]
            ref_pca_features = self.sample_pca_features.cuda()[indices]
        else:
            ref_features = self.sample_features.cuda()
            ref_pca_features = self.sample_pca_features.cuda()
        images = images.to(torch.float).to(self.device)

        _ = self.forward_modules.eval()

        batchsize,c,h,w = images.shape
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            # torch.cuda.synchronize()
            # start = time.time()
            pca_features = self.pca.transform(features)
            # torch.cuda.synchronize()
            # print(time.time() - start)
            #
            # if self.mode == 'cpu':            features = np.asarray(features)
            scales = self.ref_num_patches

            patch_scores,residuals,query_nns = self.anomaly_scorer.predict(ref_features,ref_pca_features,
                features,pca_features,topk)

            residuals = self.patch_maker.unpatch_scores(residuals,batchsize=batchsize)
            # if self. mode == 'gpu':
            residuals = residuals.reshape(batchsize, scales[0], scales[1], -1).permute(0, 3, 1, 2)


            #
            # if self.visible:
            #     ref_images = self.images[indices.cpu()]
            #     images_nns = ref_images[query_nns.cpu()].view(batchsize, -1, 3 * 8 * 8).transpose(1, 2)
            #     images_nns = torch.nn.functional.fold(images_nns, (h, w), 8, stride=8)
            #     # print(images_nns.shape)# torch.Size([8, 3,512,512])
            #     for i in range(batchsize):
            #         plt.subplot(1, 2, 1)
            #         plt.imshow(
            #             images[i].cpu().permute(1, 2, 0) * torch.tensor(IMAGENET_STD) + torch.tensor(IMAGENET_MEAN))
            #         plt.xticks([])
            #         plt.yticks([])
            #         plt.subplot(1, 2, 2)
            #         plt.xticks([])
            #         plt.yticks([])
            #         plt.imshow(
            #             images_nns[i].cpu().permute(1, 2, 0) * torch.tensor(IMAGENET_STD) + torch.tensor(IMAGENET_MEAN))
            #
            #         # title = f'global_nn={len(ids)},xy={self.distance_limit*max(self.ref_num_patches)}'
            #         # plt.title(title)
            #         plt.show()
            #         # l = [*glob('work_dirs/img/*.png')]
            #         # plt.savefig(f'work_dirs/img/{title}_{self.images.shape[0]}_{len(l)}.png')
            #         plt.close()
        if with_image_features:
            features = self.patch_maker.unpatch_scores(features, batchsize=batchsize)
            # if self. mode == 'gpu':
            features = features.reshape(batchsize, scales[0], scales[1], -1).permute(0, 3, 1, 2)
            return torch.cat([residuals,features],dim=1)
        return residuals


# Image handling classes.
class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x
