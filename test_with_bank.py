

import argparse
import os
import sys
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import torch

from torch.utils.data import DataLoader, RandomSampler

from tqdm import tqdm

import vit_encoder
from model.detectors import get_detector
from residual import backbones
from residual.matching import InterMatcher


from util.images import LabeledImagesDataset, SampleDataset
import logging

from util.metrics import compute_ap_torch, compute_pixel_auc_torch, compute_image_auc_torch, compute_pro_torch
from util.predict import predict_pixel
from util.save_code import save_dependencies_files

from util.util import fix_seed
"""
2025.8.10 updated by jingqiwu2020@163.com
"""
parser = argparse.ArgumentParser()
parser.add_argument('--data-root',type=str, default='/home/szcyxy-5/data/mvtec_anomaly_detection')
parser.add_argument('--dataset',type=str, default='bottle')
# parser.add_argument('--patchcore-patchsize', type=int,default=3)
parser.add_argument('--exp', type=int,default=1)
parser.add_argument('--batch-size', type=int,default=1)
parser.add_argument('--exp-name', type=str,default='snarm')
parser.add_argument('--detection-model', type=str,default='snarm')
parser.add_argument('--stride',type=int,default=8)
parser.add_argument('--feature-channel', type=int,default=1024)
parser.add_argument('--depths', nargs='+',type=int)
parser.add_argument("--scan",type=int,default=4)

parser.add_argument('--num-steps', type=int,default=2000)
parser.add_argument('--eval-step', type=int,default=100)

parser.add_argument('--gt-size', nargs='+',type=int,default=[392,392])

parser.add_argument('--unify',action='store_true')
parser.add_argument('--residual-method', type=str,default='concat_abs_square')
parser.add_argument('--pred-pixel',action='store_true')
# residual parameters

parser.add_argument('--patch-add-pos',type=bool, default=False,help='if add sin cos position embedding')
parser.add_argument('--target-embed-dimension',type=int, default=512)
parser.add_argument('--pos-weight',type=float, default=0)
parser.add_argument('--k-ratio',type=float, default=10000)
parser.add_argument('--topk',type=int,default=3)
parser.add_argument('--self-ratio',type=float,default=0.75)
parser.add_argument('--pca-com',type=float,default=-1)
parser.add_argument('--local-layers',nargs='+',default=['blocks.1','blocks.2','blocks.3','blocks.4','blocks.5','blocks.6','blocks.7','blocks.8'])

parser.add_argument('--patchsize',type=int,default=1)
parser.add_argument('--bank-name',type=str,default=None)
parser.add_argument('--backbone-name', type=str, default='dino')

parser.add_argument("--with-image-features",action='store_true')

parser.add_argument('--preprocess-device',type=str,default='gpu')
args = parser.parse_args()

seed = args.exp
fix_seed(seed)
gt_size = args.gt_size
print(args)


exp_name = f'{args.exp_name}_exp{args.exp}'


data_dir  = f'{args.data_root}/{args.dataset}'
work_dir = f'work_dirs/{exp_name}_{args.dataset}'
save_dependencies_files(os.path.join(work_dir, os.path.join(sys.argv[0])), args)

logging.basicConfig(filename=f'{work_dir}/log.log',level=logging.INFO)

testset = LabeledImagesDataset(f'{data_dir}/test/ng', label=1)
testset += LabeledImagesDataset(f'{data_dir}/test/ok', label=0)

print(len(testset))

test_sampler = RandomSampler(testset)

test_loader = DataLoader(testset,
                         sampler=test_sampler,
                         batch_size=args.batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with open(f'{data_dir}/input_size.txt', 'r') as f:
    h, w = [int(c) for c in f.readline().split(',')]
    image_size = (3,h,w)
# residual generator

if args.backbone_name == 'dino':
    backbone = vit_encoder.load('dinov2reg_vit_base_14')
    ref_num_patches = (h // args.stride, w // args.stride)
else:
    backbone = backbones.load(args.backbone_name)


residual_generator = InterMatcher(device, save_residuals=True, ref_num_patches=ref_num_patches,
                               add_pos_embed=args.patch_add_pos,  pos_weight=args.pos_weight, pca_com=args.pca_com)
residual_generator.load(
    backbone=backbone,
    layers_to_extract_from=args.local_layers,
    device=device,
    input_shape=image_size,
    pretrain_embed_dimension=1024,
    target_embed_dimension=args.target_embed_dimension,
    patchsize=args.patchsize,

    anomaly_scorer_num_nn=1

)
if args.unify:
    bank_path = os.path.join('work_dirs', f"{args.bank_name}.pkl")
else:
    bank_path = os.path.join(data_dir, f"{args.bank_name}.pkl")
residual_generator.fit(None,bank_path)


model=get_detector(args,image_size)
result_path = f'{work_dir}/{os.path.basename(work_dir)}_{args.eval_step}_{gt_size[0]}_{gt_size[1]}'


f = open(
    f'{result_path}_metric.csv',
    'w', encoding='utf-8')

f.write('iteration,ap,pixel_auroc,pro,image_auroc\n')
# outputsize = (h // model.out_stride, w // model.out_stride)
for iter in range(args.eval_step,args.num_steps+100,args.eval_step):
    if args.unify:
        checkpoint = torch.load(f'{work_dir.replace(args.dataset,"unify")}/iter-{iter}.pth',map_location='cpu',weights_only=True)

    else:
        checkpoint = torch.load(f'{work_dir}/iter-{iter}.pth',map_location='cpu',weights_only=True)

    msg = model.vit.load_state_dict(checkpoint,strict=False)

    print(msg)
    model.to(device)
    model.eval()
###################
    test_segs, test_gts, test_scores, test_anomaly_label = predict_pixel(test_loader, model , None,args=args,residual_generator=residual_generator)

    test_gts = test_gts.type(torch.uint8).cuda()
    test_segs = test_segs.cuda()
    try:
        from adeval import EvalAccumulatorCuda

        # determin the lower & upper bound of image-level score according to your
        #   algorithm design, ensure that the scores of 99% of the images fall within
        #   the given lower & upper bound
        score_min, score_max = torch.min(test_scores).item(), torch.max(test_scores).item()

        # also determine the lower & upper bound of values in anomap, or set to
        #   None to reuse bounds of image-level score
        anomap_min, anomap_max = torch.min(test_segs).item(), torch.max(test_segs).item()
        if score_max == score_min:
            score_max = score_max
        if anomap_max == anomap_min:
            anomap_max = anomap_max
        accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max)
        accum.add_anomap_batch(test_segs, test_gts * 255)
        accum.add_image(test_scores, test_anomaly_label.cuda())

        metrics = accum.summary()
        # print(metrics)
        ap = metrics['p_aupr']
        pro = metrics['p_aupro']
        pixel_auc = metrics['p_auroc']
        image_auc = compute_image_auc_torch(test_anomaly_label.cuda(), test_scores)
    except:
        ap = compute_ap_torch(test_gts, test_segs)
        pro = compute_pro_torch(test_gts, test_segs)
        pixel_auc = compute_pixel_auc_torch(test_gts, test_segs)
        image_auc = compute_image_auc_torch(test_anomaly_label.cuda(), test_scores)


    print(f'iter: {iter}, ap={ap},pixel_auc={pixel_auc},pro={pro},image_auc={image_auc}\n')
    f.write(f'{iter},{ap},{pixel_auc},{pro},{image_auc}\n')

    # if args.vi:
    #     save_dir = os.path.join(work_dir, 'prediction')
    #     os.makedirs(save_dir, exist_ok=True)
    #     for i in tqdm(range(test_segs.shape[0])):
    #         p = test_segs[i].cpu().numpy()
    #         path = test_loader.dataset[i]['filename'].split('/')[-1].split('.')[0]
    #         np.save(os.path.join(save_dir, path), p)