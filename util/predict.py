import time

import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm


def predict_pixel(loader,model,  knn_info, args,residual_generator=None):
    # outputsize = model.feature_size
    # print(outputsize)
    preds, gts, image_gts,image_scores = [], [], [],[]
    # if args.vote_mode:
    #     predictor = model.vote_inference
    # else:
    predictor = model.pixel_forward
    fgs = []
    latency_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch['image'].cuda()
            masks = batch['mask']
            label = batch['label']
            masks[masks >= 0.5] = 1
            masks[masks < 1] = 0

            torch.cuda.synchronize()
            start = time.time()
            if residual_generator:
                features = residual_generator._predict(images,topk=args.topk,with_image_features=args.with_image_features)
                # features = torch.tensor(features.cpu().numpy())
            else:
                features = batch['feature'].cuda()

            if hasattr(model,"with_image_head") and model.with_image_head:
                pred,image_score = predictor(features)
                image_scores.append(image_score)
            else:
                pred = predictor(features)
            torch.cuda.synchronize()
            latency_list.append(time.time()-start)


            pred = ndimage.gaussian_filter(pred.cpu(), sigma=4)

            preds.append(pred[:,0])
            gts.append(masks)
            image_gts.append(label)
    print("Latency:",(np.mean(latency_list)))
    preds = torch.tensor(np.concatenate(preds,axis=0)) # 132 1024

    gts = torch.cat(gts).squeeze()  # 132,256,256
    image_gts = torch.cat(image_gts)

    preds = preds.cuda()

    if len(image_scores) == 0:
        image_scores = torch.max(torch.nn.functional.avg_pool2d(preds,16,stride=2).view(preds.shape[0], -1), dim=-1)[0]
    else:
        image_scores = torch.cat(image_scores,dim=0)
        
    return preds, gts, image_scores, image_gts

