import math
from typing import Optional

import torch
import torch.nn as nn
from mamba_ssm import Mamba
from torch import Tensor

from model.util import prepocess_residual

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def get_norm_layer(norm: str):
    norm = {
        "BN": nn.BatchNorm2d,
        "LN": nn.LayerNorm,
        "GN": nn.GroupNorm,
    }[norm.upper()]
    return norm


def get_act_layer(act: str):
    act = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "swish": nn.SiLU,
        "mish": nn.Mish,
        "leaky_relu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
        "gelu": nn.GELU,
    }[act.lower()]
    return act


class ConvNormAct2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding="same",
        dilation=1,
        groups=1,
        conv_kwargs=None,
        norm_layer=None,
        norm_kwargs=None,
        act_layer=None,
        act_kwargs=None,
    ):
        super(ConvNormAct2d, self).__init__()

        conv_kwargs = {}
        if norm_layer:
            conv_kwargs["bias"] = False
        if padding == "same" and stride > 1:
            # if kernel_size is even, -1 is must
            padding = (kernel_size - 1) // 2

        self.conv = self._build_conv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            conv_kwargs,
        )
        self.norm = None


        if norm_layer:
            norm_kwargs = {}
            if norm_layer == "GN":
                self.norm = get_norm_layer(norm_layer)(
                    32,out_channels
                )
            else:
                self.norm = get_norm_layer(norm_layer)(
                out_channels, **norm_kwargs
            )
        self.act = None
        if act_layer:
            act_kwargs = {}
            self.act = get_act_layer(act_layer)(**act_kwargs)

    def _build_conv(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        conv_kwargs,
    ):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            **conv_kwargs,
        )

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            if isinstance(self.norm, nn.LayerNorm):
                b,c,h,w = x.shape
                x = self.norm(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, c, h, w)
            else:
                x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x



class SelfNavigatedMambaBlock(nn.Module):
    def __init__(
        self, dim,  norm_cls=RMSNorm, fused_add_norm=True, residual_in_fp32=True,

    ):

        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.mixer = Mamba(
                            # This module uses roughly 3 * expand * d_model^2 parameters
                            d_model=dim,  # Model dimension d_model
                            d_state=4,  # SSM state expansion factor
                            d_conv=4,  # Local convolution width
                            expand=2,  # Block expansion factor
                        )
        self.norm = norm_cls(dim)

        self.conv = nn.Conv2d(dim,dim,3,1,1)

    def forward(
        self, hidden_states: Tensor,mask, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        B,L,C = hidden_states.shape
        L_sqrt = round(math.sqrt(L))
        hidden_states = hidden_states.permute(0,2,1).reshape(B,C,L_sqrt,L_sqrt)
        hidden_states = self.conv(hidden_states).flatten(2,3).permute(0,2,1)
        easy_hidden_state = hidden_states[mask == 0].reshape(B, -1, C)
        # hard_hidden_state = hidden_states[mask == 1].reshape(B, -1, C)
        easy_hidden_state = self.mixer(easy_hidden_state,inference_params)
        # hard_hidden_state = self.mixer2(hard_hidden_state)
        x = hidden_states.clone()
        x[mask == 0] =easy_hidden_state.flatten(0, 1)
        return x, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
class SelfNavigatedMamba(nn.Module):

    def __init__(self, stride=8,
                 residual_method='square',drop_out_rate=0.1,self_ratio=0.5,scan=1,scale_list= [3,6,12,24],
                 **kwargs):
        super().__init__()
        self.stride = stride
        self.patch_size = kwargs['patch_size']
        self.img_size = kwargs['img_size']
        self.residual_method = residual_method
        self.scan=scan
        self.scale_list = scale_list
        print(f'model residual:{self.residual_method}')
        self.online_bank = None

        self.self_ratio = self_ratio
        self.simple_predictor = nn.Conv2d(kwargs['in_chans'] // 2, 1, kernel_size=1, bias=True)


        print(f'Self_ratio: {self_ratio},scan:{scan}, Mamba depth: {kwargs["depth"]}')
        self.conv_list = nn.ModuleList()
        self.mamba_list = nn.ModuleList()
        self.drop_out_list = nn.ModuleList()
        self.norm_list = nn.ModuleList()
        self.head_list = nn.ModuleList()
        for atrous_rate in scale_list:
            self.conv_list.append(nn.Conv2d(kwargs['in_chans'], kwargs['embed_dim'],
                                            kernel_size=1,
                                            padding=0,
                                            bias=True,
                                            ))
            self.mamba_list.append(torch.nn.Sequential(*[
                    SelfNavigatedMambaBlock(dim=kwargs['embed_dim']) for i in range(kwargs['depth'])]))

            self.drop_out_list.append(nn.Dropout(drop_out_rate))
            self.norm_list.append(nn.LayerNorm(kwargs['embed_dim']))
            for s in range(self.scan):
                conv_norm_act = ConvNormAct2d
                self.head_list.append(torch.nn.Sequential(
                    conv_norm_act(
                        in_channels=kwargs['embed_dim'],
                        out_channels=kwargs['embed_dim'],
                        kernel_size=1 if atrous_rate == 1 else 3,
                        padding=0 if atrous_rate == 1 else atrous_rate,
                        dilation=atrous_rate,
                        norm_layer='BN',
                        act_layer="RELU",
                    ),
                    nn.Conv2d(kwargs['embed_dim'],kwargs['num_classes'],kernel_size=1)
                )
                )

    def forward_features(self, x, scan0_mask,branch_id):

        fea = self.conv_list[branch_id](x)

        B, C, H, W = fea.shape
        y = x.new_empty((self.scan , B, H * W, C))

        if self.scan == 4:
            reshape_mask = scan0_mask.reshape(B, H, W)
            scan1_mask = reshape_mask.transpose(1, 2).flatten(1, 2)
            scan2_mask = torch.flip(scan0_mask, dims=[1])
            scan3_mask = torch.flip(scan1_mask, dims=[1])
            all_mask = torch.cat([scan0_mask, scan1_mask, scan2_mask, scan3_mask], dim=0)

            y[0, :, :, :] = fea.flatten(2, 3).transpose(1, 2)
            y[1, :, :, :] = fea.transpose(dim0=2, dim1=3).flatten(2, 3).transpose(1, 2)
            y[2:4, :, :, :] = torch.flip(y[0:2, :, :, :], dims=[-2])

        else:
            y =  fea.flatten(2, 3).transpose(1, 2).unsqueeze(0)

            all_mask = scan0_mask
        hidden_state = y.flatten(0,1).clone()

        residual= None
        for layer in self.mamba_list[branch_id]:

            hidden_state,residual = layer(hidden_state,all_mask,residual)

        hidden_state = self.norm_list[branch_id](
            self.drop_out_list[branch_id](hidden_state) + residual)

        y = hidden_state.reshape(self.scan, B, H * W, C)

        if self.scan == 4:
            y[1] = y[1].reshape(B, -1, H, C).transpose(1, 2).flatten(1, 2)
            y[2] = y[2].flip(dims=[1])
            y[3] = y[3].flip(dims=[1]).reshape(B, -1, H, C).transpose(1, 2).flatten(1, 2)
        return y

    def forward(self, x,head_id = None ):
        residual,image_feature = torch.chunk(x,2,1)
        residual = prepocess_residual(residual, self.residual_method)

        simple_pred = self.simple_predictor(residual)
        simple_pred = torch.nn.functional.sigmoid(simple_pred)

        filter_scores = simple_pred + residual.mean(1, keepdim=True)
        # print(residual.mean(1, keepdim=True).shape)

        b,c,h,w = image_feature.shape
        sample_num = round(h*w*self.self_ratio)

        _, indices = torch.topk(filter_scores.flatten(1), k=sample_num, dim=1, largest=False)
        image_feature = image_feature.permute(0,2,3,1).flatten(1,2)
        self_residual = []
        for i in range(b):
            online_bank = image_feature[i][indices[i]]
            if self.online_bank is not None:
                online_bank = torch.cat((online_bank,self.online_bank),dim=0)

            distance = torch.cdist(image_feature[i],online_bank,p=2)
            filter_distance = distance
            filter_distance[indices[i],torch.arange(indices.shape[1])] = torch.inf
            matched_dist,nns = torch.topk(filter_distance,dim=1,k=1,largest=False)
            self_residual.append((online_bank[nns[:,0]] - image_feature[i]).unsqueeze(0))

        self_residual = torch.cat(self_residual,dim=0)
        self_residual = self_residual.reshape(b,h,w,c).permute(0,3,1,2)

        self_residual = prepocess_residual(self_residual,self.residual_method)

        x = torch.cat((residual,self_residual),dim=1)
        mask = torch.zeros((b, h * w))
        mask[torch.arange(indices.shape[0]).unsqueeze(1), indices] = 1
        # x = self.forward_features(x, mask)
        # x = x.permute(0,2,1).reshape(b,-1,h,w)
        logits_list = []
        if head_id is not None:
            f = self.forward_features(x, mask, head_id)
            f = f.permute(0,  1,3,2).reshape(self.scan,b, -1, h, w)
            for s in range(self.scan):
                logits_list.append(self.head_list[head_id*self.scan+s](f[s]))

            return simple_pred, logits_list
        for branch_id in range(len(self.scale_list)):
            f = self.forward_features(x,mask,branch_id)
            f = f.permute(0, 1, 3, 2).reshape(self.scan, b, -1, h, w)
            for s in range(self.scan):
                logits_list.append(self.head_list[branch_id* self.scan + s](f[s]))

        return simple_pred, logits_list
