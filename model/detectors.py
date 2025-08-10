import torch
from torch import nn


from model.mamba_util import SelfNavigatedMamba


class SelfNavigatedResidualMambaDecoder(nn.Module):
    def __init__(self, image_size, stride=8, patch_size=(1, 1), residual_method='square',
                in_chans=1024, num_classes=2, embed_dim=128, depth=1,scan=4,atrous_rates=[1,12],self_ratio=0.5,num_extra_branch=0):
        super(SelfNavigatedResidualMambaDecoder, self).__init__()
        self.image_size = image_size
        feature_size = (image_size[1] // stride, image_size[2] // stride)

        self.patch_size = patch_size
        self.feature_size = feature_size

        self.out_stride = stride  # //len(depths)


        # feature_size = (window_size*math.floor(feature_size[0]/window_size),window_size*math.floor(feature_size[1]/window_size))
        self.vit_img_size = (
            image_size[1] // self.out_stride * patch_size[0], image_size[2] // self.out_stride * patch_size[1])
        self.vit = SelfNavigatedMamba(img_size=self.vit_img_size, in_chans=in_chans,
                                         num_classes=num_classes,
                                         patch_size=patch_size, embed_dim=embed_dim,
                                         residual_method=residual_method, depth=depth,scan=scan,atrous_rates=atrous_rates,self_ratio=self_ratio,num_extra_branch=num_extra_branch)

    def pixel_forward(self, feature_residual, img_gts=None, if_train=False, args=None,
                 **kwargs):
        feature_residual = feature_residual.to('cuda', non_blocking=True)
        if img_gts is not None:
            masks = img_gts
            masks = masks.to('cuda', non_blocking=True)

        selector_pred, fl_logits = self.vit(feature_residual)
        num_heads = len(fl_logits)
        fl_logits = torch.cat(fl_logits, dim=0)

        fl_logits = torch.nn.functional.interpolate(fl_logits, (self.image_size[1], self.image_size[2]),
                                                    mode='bilinear',
                                                    align_corners=True)


        fl_logits = torch.sigmoid(fl_logits)


        fl_logits = fl_logits.reshape(num_heads, -1, *fl_logits.shape[1:]).mean(dim=0)


        return fl_logits
def get_detector(args, image_size, lr=None):
    if args.detection_model == 'snarm':
        model = SelfNavigatedResidualMambaDecoder(image_size, stride=args.stride, patch_size=(1, 1),
                                              residual_method=args.residual_method,
                                              in_chans=args.feature_channel,
                                              num_classes=1, embed_dim=128, depth=args.depths[0],
                                     scan=args.scan,self_ratio=args.self_ratio)

    if lr is None: return model

    optimizer = torch.optim.AdamW(model.vit.parameters(), lr=lr, weight_decay=args.weight_decay)
    return model, optimizer