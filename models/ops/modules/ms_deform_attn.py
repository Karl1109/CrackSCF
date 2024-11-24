# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LowRankLinear, self).__init__()
        # rank 是低秩近似的秩，通常远小于 in_features 和 out_features
        self.rank = rank

        # 分解原始权重矩阵为两个矩阵的乘积
        self.linear1 = nn.Linear(in_features, rank)
        self.linear2 = nn.Linear(rank, out_features)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension 维度是256
        :param n_levels     number of feature levels 4层scale
        :param n_heads      number of attention heads 8头注意力
        :param n_points     number of sampling points per attention head per feature level 4个点
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.sampling_offsets = LowRankLinear(d_model, n_heads * n_levels * n_points * 2, 4)
        # self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.attention_weights = LowRankLinear(d_model, n_heads * n_levels * n_points, 4)
        # self.value_proj = nn.Linear(d_model, d_model)
        self.value_proj = LowRankLinear(d_model, d_model, 4)
        # self.output_proj = nn.Linear(d_model, d_model)
        self.output_proj = LowRankLinear(d_model, d_model, 4)
        # self.output_proj = self.value_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        # constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        # constant_(self.attention_weights.weight.data, 0.)
        # constant_(self.attention_weights.bias.data, 0.)
        # xavier_uniform_(self.value_proj.weight.data)
        # constant_(self.value_proj.bias.data, 0.)
        # xavier_uniform_(self.output_proj.weight.data)
        # constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """

        """
        query是上一层的输出加上了位置编码
        :param query                       (N, Length_{query}, C)
        
        参考点位的坐标
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        encoder是上一层的输出，decoder使用的是encoder的输出 [bs, all hw, 256]
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        4个特征层的高宽
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        各个特征层的起始index的下标 如: [    0,  8056, 10070, 10583]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        [bs,all hw]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """

        # query 是src+pos，query下面变成了attention_weights
        # input_flatten 是src，input_flatten 对应了V
        # bs, all hw（decoder 是300）, 256
        # (1, 8000, 256)
        N, Len_q, _ = query.shape
        # (1, 8000, 256)
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        # 对encoder上一层的输出，或者decoder使用的encoder的输出 进行一层全连接变换，channel不变
        # (1, 8000, 256)
        value = self.value_proj(input_flatten)

        if input_padding_mask is not None:
            # 填充0 [bs, all hw,256]
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        # 分成多头，拆分的是最后的256 [bs,all hw,256] -> [bs,all hw, 8, 32]
        # 分成了8头，(1, 8000, 8, 32)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)

        # 偏移量，一共有四个偏移点
        # like ( bs, all hw, 8, 4, 4, 2 ) 8个头，4个特征层，4个采样点 2个偏移量坐标
        # (1, 8000, 8, 4, 4, 2)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)

        # 由 query 获取的attention，形状是和 value 一样的
        # (1, 8000, 8, 4, 4, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        # 执行 elif
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output
