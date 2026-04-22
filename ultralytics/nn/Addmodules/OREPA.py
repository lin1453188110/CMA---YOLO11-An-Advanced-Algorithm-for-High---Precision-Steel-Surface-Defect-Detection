"""
OREPA.py - 在线重参数化高效聚合与小波变换模块
=============================================

主要功能：
1. WTConv2d: 小波卷积层，利用小波变换在频域进行特征提取
2. LSKA: 大核空间注意力，使用深度可分离卷积近似大核卷积
3. DCNv3_pytorch: 可变形卷积v3的PyTorch实现
4. OREPA: 在线重参数化高效聚合卷积
5. CSOTC/C3k2_OREPA_neck: 复合瓶颈结构

技术原理：
- 小波变换：多尺度频域分析，捕获不同频率的特征
- 重参数化：训练时多分支，部署时单分支，平衡效率和精度
- 可变形卷积：学习采样位置偏移，适应几何变换

作者: CSDN:Snu77
参考论文: 
- OREPA: Online Convolutional Re-parameterization (https://arxiv.org/abs/2204.00826)
- WTConv: Wavelet Convolutions (https://arxiv.org/abs/2407.10948)
"""

import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
from einops import rearrange
from torch.nn.init import xavier_uniform_, constant_
from functools import partial
import pywt
import pywt.data

# 导出模块列表
__all__ = ['OREPA', 'CSOTC', 'C3k2_OREPA_neck']


# ==================== 小波变换相关函数 ====================

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    """
    创建小波滤波器
    
    功能：根据小波类型创建分解和重构滤波器
    
    参数:
    - wave: 小波类型，如'db1', 'haar', 'sym2'等
    - in_size: 输入通道数
    - out_size: 输出通道数
    - type: 数据类型
    
    返回:
    - dec_filters: 分解滤波器
    - rec_filters: 重构滤波器
    """
    # 获取小波对象
    w = pywt.Wavelet(wave)
    
    # 分解滤波器（翻转顺序）
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)  # 高通分解滤波器
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)  # 低通分解滤波器
    
    # 构建二维分解滤波器（4个方向的组合）
    # LL: dec_lo * dec_lo (低频)
    # LH: dec_lo * dec_hi (水平高频)
    # HL: dec_hi * dec_lo (垂直高频)
    # HH: dec_hi * dec_hi (对角高频)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    # 复制滤波器以适配多通道
    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    # 重构滤波器（翻转顺序）
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    
    # 构建二维重构滤波器
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    # 复制滤波器以适配多通道
    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    """
    小波变换（分解）
    
    功能：对输入进行小波分解
    
    参数:
    - x: 输入张量 [B, C, H, W]
    - filters: 小波滤波器
    
    返回:
    - x: 小波系数，形状 [B, C, 4, H//2, W//2]
    """
    b, c, h, w = x.shape
    # 计算填充量
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    
    # 步长为2的分组卷积，实现下采样
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    
    # 重组形状：[B, C*4, H//2, W//2] -> [B, C, 4, H//2, W//2]
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    """
    逆小波变换（重构）
    
    功能：从小波系数重构图像
    
    参数:
    - x: 小波系数 [B, C, 4, H//2, W//2]
    - filters: 小波滤波器
    
    返回:
    - x: 重构图像 [B, C, H, W]
    """
    b, c, _, h_half, w_half = x.shape
    
    # 计算填充量
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    
    # 重组形状：[B, C, 4, H//2, W//2] -> [B, C*4, H//2, W//2]
    x = x.reshape(b, c * 4, h_half, w_half)
    
    # 步长为2的转置卷积，实现上采样
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


# ==================== 小波卷积层 ====================

class WTConv2d(nn.Module):
    """
    小波卷积层 (Wavelet Transform Convolution)
    
    功能说明：
    - 利用小波变换在频域进行特征提取
    - 多级小波分解捕获不同频率的特征
    - 高频分支学习小波系数，低频分支使用标准卷积
    
    参数:
    - in_channels: 输入通道数
    - out_channels: 输出通道数（必须等于输入通道数）
    - kernel_size: 卷积核大小
    - stride: 步长
    - bias: 是否使用偏置
    - wt_levels: 小波变换的层数
    - wt_type: 小波类型，如'db1'
    
    工作流程：
    1. 多级小波分解：逐级提取高频和低频成分
    2. 高频处理：每级高频通过卷积处理
    3. 逆变换重构：逐级逆变换还原分辨率
    4. 残差连接：主分支与小波分支融合
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        """初始化小波卷积层"""
        super(WTConv2d, self).__init__()

        # 确保输入输出通道相等
        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels  # 小波变换层数
        self.stride = stride
        self.dilation = 1

        # 创建小波滤波器（分解和重构）
        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        # 绑定小波变换函数
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        # 基础卷积分支：标准深度卷积
        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        # 小波卷积分支列表
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)])
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)])

        # 步长处理（如果stride > 1）
        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):
        """
        前向传播
        
        参数:
        - x: 输入张量 [B, C, H, W]
        
        返回:
        - x: 输出张量 [B, C, H//stride, W//stride]
        """
        # 用于存储各级的小波系数
        x_ll_in_levels = []  # 低频成分列表
        x_h_in_levels = []   # 高频成分列表
        shapes_in_levels = []  # 各级的shape

        # 当前低频分量
        curr_x_ll = x

        # ========== 前向：小波分解 ==========
        for i in range(self.wt_levels):
            # 记录当前shape
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            
            # 处理奇数尺寸的padding
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            # 执行一级小波分解
            curr_x = self.wt_function(curr_x_ll)
            
            # 提取低频分量（第一通道）
            curr_x_ll = curr_x[:, :, 0, :, :]

            # 处理高频分量
            shape_x = curr_x.shape
            # 重组：[B, C*4, H//2, W//2] -> [B, C, 4, H//2, W//2]
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            # 卷积处理高频分量
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            # 重组回原始形状
            curr_x_tag = curr_x_tag.reshape(shape_x)

            # 保存各级系数
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])  # 低频
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])  # 高频（3个方向）

        # ========== 逆向：小波重构 ==========
        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            # 取出当前级的系数
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            # 残差连接
            curr_x_ll = curr_x_ll + next_x_ll

            # 拼接低频和高频
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            
            # 逆小波变换
            next_x_ll = self.iwt_function(curr_x)

            # 裁剪到目标尺寸
            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        # ========== 融合主分支和小波分支 ==========
        x = self.base_scale(self.base_conv(x))  # 主分支：标准卷积
        x = x + x_tag  # 小波分支

        # 处理步长
        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    """
    缩放模块
    
    功能：对输入进行逐元素缩放
    
    参数:
    - dims: 缩放因子的维度
    - init_scale: 初始缩放值
    - init_bias: 初始偏置
    """
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        """逐元素乘法"""
        return torch.mul(self.weight, x)


# ==================== 大核空间注意力 ====================

class LSKA(nn.Module):
    """
    大核空间注意力 (Large-Kernel Spatial Attention)
    
    功能说明：
    - 使用深度可分离卷积近似大核卷积
    - 捕获大范围的空間上下文信息
    - 比标准大核卷积更高效，参数量更少
    
    参数:
    - dim: 通道数
    - k_size: 目标大核大小，支持7, 11, 23, 35, 41, 53
    
    实现方式：
    - 使用多个小核卷积的组合来近似大核效果
    - 0h/0v: 1x3/3x1卷积提取基本特征
    - spatial_h/spatial_v: 带膨胀的1x3/3x1卷积扩展感受野
    - conv1: 1x1卷积整合信息
    """
    
    def __init__(self, dim, k_size=7):
        super().__init__()

        self.k_size = k_size

        # 根据目标核大小选择不同的卷积配置
        if k_size == 7:
            # 小核组合：3x3 + 膨胀3x3 ≈ 7x7
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
            # 膨胀卷积：dilation=2，感受野进一步扩大
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, 2), groups=dim,
                                            dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=(2, 0), groups=dim,
                                            dilation=2)
        elif k_size == 11:
            # 3x3 + 膨胀5x5 ≈ 11x11
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 3), stride=(1, 1), padding=(0, (3 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(3, 1), stride=(1, 1), padding=((3 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, 4), groups=dim,
                                            dilation=2)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), groups=dim,
                                            dilation=2)
        elif k_size == 23:
            # 5x5 + 膨胀7x7 ≈ 23x23
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 7), stride=(1, 1), padding=(0, 9), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(7, 1), stride=(1, 1), padding=(9, 0), groups=dim,
                                            dilation=3)
        elif k_size == 35:
            # 5x5 + 膨胀11x11 ≈ 35x35
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 11), stride=(1, 1), padding=(0, 15), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(11, 1), stride=(1, 1), padding=(15, 0), groups=dim,
                                            dilation=3)
        elif k_size == 41:
            # 5x5 + 膨胀13x13 ≈ 41x41
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 13), stride=(1, 1), padding=(0, 18), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(13, 1), stride=(1, 1), padding=(18, 0), groups=dim,
                                            dilation=3)
        elif k_size == 53:
            # 5x5 + 膨胀17x17 ≈ 53x53
            self.conv0h = nn.Conv2d(dim, dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5 - 1) // 2), groups=dim)
            self.conv0v = nn.Conv2d(dim, dim, kernel_size=(5, 1), stride=(1, 1), padding=((5 - 1) // 2, 0), groups=dim)
            self.conv_spatial_h = nn.Conv2d(dim, dim, kernel_size=(1, 17), stride=(1, 1), padding=(0, 24), groups=dim,
                                            dilation=3)
            self.conv_spatial_v = nn.Conv2d(dim, dim, kernel_size=(17, 1), stride=(1, 1), padding=(24, 0), groups=dim,
                                            dilation=3)

        # 1x1卷积整合信息
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        """
        前向传播
        
        参数:
        - x: 输入张量 [B, C, H, W]
        
        返回:
        - 输出：u * attn [B, C, H, W]
        """
        u = x.clone()  # 保存输入用于残差
        # 串行应用四个卷积层
        attn = self.conv0h(x)
        attn = self.conv0v(attn)
        attn = self.conv_spatial_h(attn)
        attn = self.conv_spatial_v(attn)
        attn = self.conv1(attn)
        return u * attn  # 通道乘法融合


# ==================== DCNv3 相关函数 ====================

def _get_reference_points(spatial_shapes, device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h=0, pad_w=0,
                          stride_h=1, stride_w=1):
    """
    生成参考点坐标
    
    功能：为可变形卷积生成采样位置的参考点
    
    参数:
    - spatial_shapes: 空间形状 (B, H, W, _)
    - device: 设备
    - kernel_h/kernel_w: 卷积核尺寸
    - dilation_h/dilation_w: 膨胀率
    - pad_h/pad_w: 填充
    - stride_h/stride_w: 步长
    
    返回:
    - ref: 参考点坐标 [1, H_out, W_out, 1, 2]
    """
    _, H_, W_, _ = spatial_shapes
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    # 生成网格点
    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            (dilation_h * (kernel_h - 1)) // 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (H_out - 1) * stride_h,
            H_out,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (W_out - 1) * stride_w,
            W_out,
            dtype=torch.float32,
            device=device))

    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_

    ref = torch.stack((ref_x, ref_y), -1).reshape(
        1, H_out, W_out, 1, 2)

    return ref


def _generate_dilation_grids(spatial_shapes, kernel_h, kernel_w, dilation_h, dilation_w, group, device):
    """
    生成膨胀网格
    
    功能：为每个组生成膨胀采样的偏移网格
    """
    _, H_, W_, _ = spatial_shapes
    points_list = []
    x, y = torch.meshgrid(
        torch.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) +
            (kernel_w - 1) * dilation_w, kernel_w,
            dtype=torch.float32,
            device=device),
        torch.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) +
            (kernel_h - 1) * dilation_h, kernel_h,
            dtype=torch.float32,
            device=device))

    points_list.extend([x / W_, y / H_])
    grid = torch.stack(points_list, -1).reshape(-1, 1, 2). \
        repeat(1, group, 1).permute(1, 0, 2)
    grid = grid.reshape(1, 1, 1, group * kernel_h * kernel_w, 2)

    return grid


def dcnv3_core_pytorch(input, offset, mask, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, group, group_channels, offset_scale):
    """
    DCNv3核心操作的PyTorch实现（仅用于调试）
    
    功能：实现可变形卷积v3的核心采样逻辑
    
    参数:
    - input: 输入特征 [N_, H_in, W_in, C]
    - offset: 位置偏移 [N_, H_out, W_out, group*P_*2]
    - mask: 注意力掩码 [N_, H_out, W_out, group*P_]
    - kernel_h/kernel_w: 卷积核尺寸
    - stride_h/stride_w: 步长
    - pad_h/pad_w: 填充
    - dilation_h/dilation_w: 膨胀率
    - group: 分组数
    - group_channels: 每组的通道数
    - offset_scale: 偏移缩放因子
    
    返回:
    - output: 采样后的特征 [N_, H_out, W_out, C]
    """
    # 填充输入
    input = F.pad(input, [0, 0, pad_h, pad_h, pad_w, pad_w])
    N_, H_in, W_in, _ = input.shape
    _, H_out, W_out, _ = offset.shape

    # 获取参考点
    ref = _get_reference_points(input.shape, input.device, kernel_h, kernel_w, dilation_h, dilation_w, pad_h, pad_w, stride_h, stride_w)
    # 生成膨胀网格
    grid = _generate_dilation_grids(input.shape, kernel_h, kernel_w, dilation_h, dilation_w, group, input.device)
    # 空间归一化
    spatial_norm = torch.tensor([W_in, H_in]).reshape(1, 1, 1, 2).repeat(1, 1, 1, group * kernel_h * kernel_w).to(input.device)

    # 计算采样位置
    sampling_locations = (ref + grid * offset_scale).repeat(N_, 1, 1, 1, 1).flatten(3, 4) + offset * offset_scale / spatial_norm

    P_ = kernel_h * kernel_w
    sampling_grids = 2 * sampling_locations - 1
    
    # 输入变形
    input_ = input.view(N_, H_in * W_in, group * group_channels).transpose(1, 2).reshape(N_ * group, group_channels, H_in, W_in)
    sampling_grid_ = sampling_grids.view(N_, H_out * W_out, group, P_, 2).transpose(1, 2).flatten(0, 1)
    # 双线性插值采样
    sampling_input_ = F.grid_sample(input_, sampling_grid_, mode='bilinear', padding_mode='zeros', align_corners=False)

    # 应用掩码
    mask = mask.view(N_, H_out * W_out, group, P_).transpose(1, 2).reshape(N_ * group, 1, H_out * W_out, P_)
    output = (sampling_input_ * mask).sum(-1).view(N_, group * group_channels, H_out * W_out)

    return output.transpose(1, 2).reshape(N_, H_out, W_out, -1).contiguous()


# ==================== 辅助模块 ====================

class to_channels_first(nn.Module):
    """将张量从通道后置格式转为通道前置格式"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # [B, H, W, C] -> [B, C, H, W]
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):
    """将张量从通道前置格式转为通道后置格式"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # [B, C, H, W] -> [B, H, W, C]
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim, norm_layer, in_format='channels_last', out_format='channels_last', eps=1e-6):
    """
    构建归一化层
    
    功能：根据输入输出格式构建适当的归一化层
    
    参数:
    - dim: 特征维度
    - norm_layer: 归一化类型 ('BN' 或 'LN')
    - in_format: 输入格式
    - out_format: 输出格式
    """
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    """构建激活函数层"""
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()
    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


def _is_power_of_2(n):
    """检查是否为2的幂"""
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")
    return (n & (n - 1) == 0) and n != 0


class CenterFeatureScaleModule(nn.Module):
    """
    中心特征缩放模块
    
    功能：计算中心特征的缩放因子，用于DCNv3中的特征融合
    """
    def forward(self, query, center_feature_scale_proj_weight, center_feature_scale_proj_bias):
        center_feature_scale = F.linear(query,
                                        weight=center_feature_scale_proj_weight,
                                        bias=center_feature_scale_proj_bias).sigmoid()
        return center_feature_scale


# ==================== DCNv3 ====================

class DCNv3_pytorch(nn.Module):
    """
    可变形卷积v3 (Deformable Convolution v3) PyTorch实现
    
    功能说明：
    - 通过学习采样位置的偏移量来适应几何变换
    - 使用分组机制提高效率
    - 包含中心特征缩放机制增强特征表达
    
    参数:
    - channels: 通道数
    - kernel_size: 卷积核大小
    - stride: 步长
    - pad: 填充
    - dilation: 膨胀率
    - group: 分组数
    - offset_scale: 偏移缩放因子
    - act_layer: 激活函数类型
    - norm_layer: 归一化层类型
    - center_feature_scale: 是否使用中心特征缩放
    
    工作流程：
    1. 深度卷积提取空间信息
    2. 预测偏移量(offset)和注意力掩码(mask)
    3. 基于偏移量和掩码进行可变形采样
    4. 可选的中心特征缩放融合
    """
    
    def __init__(self, channels=64, kernel_size=3, dw_kernel_size=None, stride=1, pad=1, dilation=1,
                 group=4, offset_scale=1.0, act_layer='GELU', norm_layer='LN', center_feature_scale=False):
        """初始化DCNv3模块"""
        super().__init__()

        # 参数校验
        if channels % group != 0:
            raise ValueError(f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group
        
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        
        # 警告：建议_d_per_group为2的幂以提高CUDA实现效率
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DCNv3 to make the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation.")
        
        # 保存参数
        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.center_feature_scale = center_feature_scale
        
        # 深度卷积分支：提取空间信息用于预测偏移
        self.dw_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=dw_kernel_size, stride=1,
                      padding=(dw_kernel_size - 1) // 2, groups=channels),
            build_norm_layer(channels, norm_layer, 'channels_first', 'channels_last'),
            build_act_layer(act_layer))
        
        # 偏移预测头：预测采样位置的偏移量
        # 输出：group * kernel_size * kernel_size * 2 (x和y偏移)
        self.offset = nn.Linear(channels, group * kernel_size * kernel_size * 2)
        
        # 掩码预测头：预测每个采样点的重要性权重
        self.mask = nn.Linear(channels, group * kernel_size * kernel_size)
        
        # 输入输出投影
        self.input_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        
        # 初始化参数
        self._reset_parameters()
        
        # 中心特征缩放模块
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(torch.zeros((group, channels), dtype=torch.float))
            self.center_feature_scale_proj_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float).view((1,)).repeat(group,)))
            self.center_feature_scale_module = CenterFeatureScaleModule()

    def _reset_parameters(self):
        """初始化网络参数"""
        constant_(self.offset.weight.data, 0.)
        constant_(self.offset.bias.data, 0.)
        constant_(self.mask.weight.data, 0.)
        constant_(self.mask.bias.data, 0.)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        """
        前向传播
        
        参数:
        - input: 输入张量 [B, C, H, W]
        
        返回:
        - output: 输出张量 [B, C, H/stride, W/stride]
        """
        # 转换为通道后置格式
        input = input.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        N, H, W, _ = input.shape
        
        # 输入投影
        x = self.input_proj(input)
        x_proj = x
        
        # ========== 深度卷积提取空间信息 ==========
        x1 = input.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        x1 = self.dw_conv(x1)  # 深度卷积
        
        # ========== 预测偏移和掩码 ==========
        offset = self.offset(x1)  # [B, H, W, group*k*k*2]
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)  # [B, H, W, group, k*k]
        mask = F.softmax(mask, -1).reshape(N, H, W, -1)  # softmax归一化
        
        # ========== 可变形采样 ==========
        x = dcnv3_core_pytorch(
            x, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale)
        
        # ========== 中心特征缩放 ==========
        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # 调整形状并进行特征融合
            center_feature_scale = center_feature_scale[..., None].repeat(1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        
        # ========== 输出投影 ==========
        x = self.output_proj(x).permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        return x


class DeformConv(nn.Module):
    """
    可变形卷积封装
    
    功能：对DCNv3进行封装，提供更简洁的接口
    """
    def __init__(self, in_channels, groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()
        self.deform_conv = DCNv3_pytorch(in_channels)

    def forward(self, x):
        out = self.deform_conv(x)
        return out


class deformable_LKA(nn.Module):
    """
    可变形大核注意力
    
    功能：结合可变形卷积和大核注意力的优势
    """
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv(dim, kernel_size=(5, 5), padding=2, groups=dim)
        self.conv_spatial = DeformConv(dim, kernel_size=(7, 7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class deformable_LKA_Attention(nn.Module):
    """
    可变形大核注意力模块
    
    功能：完整的可变形LKA注意力模块，包含投影层和门控机制
    """
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = deformable_LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut  # 残差连接
        return x


# ==================== RFAConv ====================

class RFAConv(nn.Module):
    """
    感受野注意力卷积 (Receptive Field Attention Convolution)
    
    功能：
    - 通过加权聚合多个卷积核的感受野信息
    - 动态学习不同空间位置的权重
    - 增强特征的多尺度表达能力
    """
    
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        # 获取卷积权重的分支
        self.get_weight = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1,
                      groups=in_channel, bias=False))
        
        # 生成特征的分支
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=kernel_size, p=0)

    def forward(self, x):
        b, c = x.shape[0:2]
        # 获取注意力权重
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        # 归一化权重
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)
        # 生成特征
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h, w)
        # 加权聚合
        weighted_data = feature * weighted
        # 重组为常规卷积格式
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size, n2=self.kernel_size)
        return self.conv(conv_data)


# ==================== 多尺度空洞注意力（重复定义） ====================

class DilateAttention(nn.Module):
    """空洞注意力 - 见MSDA.py"""

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        B, d, H, W = q.shape
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)
        k = self.unfold(k).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2, 3)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        v = self.unfold(v).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3, 2)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class MultiDilatelocalAttention(nn.Module):
    """多空洞局部注意力 - 见MSDA.py"""

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3, 4]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        y = x.clone()
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        y1 = y.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2)
        for i in range(self.num_dilation):
            y1[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])
        y2 = y1.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        y3 = self.proj(y2)
        y4 = self.proj_drop(y3).permute(0, 3, 1, 2)
        return y4


# ==================== 辅助函数 ====================

def autopad(k, p=None, d=1):
    """自动填充 - 见MSDA.py"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """标准卷积 - 见MSDA.py"""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


def transI_fusebn(kernel, bn):
    """
    融合卷积核和BatchNorm
    
    功能：将卷积层的权重和BN层融合为一个等效的卷积层
    
    参数:
    - kernel: 卷积核权重
    - bn: BatchNorm层
    
    返回:
    - 融合后的卷积核和偏置
    """
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std


def transVI_multiscale(kernel, target_kernel_size):
    """
    多尺度卷积核变换
    
    功能：将卷积核填充到目标尺寸
    """
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    return F.pad(kernel, [W_pixels_to_pad, W_pixels_to_pad, H_pixels_to_pad, H_pixels_to_pad])


# ==================== OREPA ====================

class OREPA(nn.Module):
    """
    在线重参数化高效聚合卷积 (Online Convolutional Re-parameterization)
    
    功能说明：
    - 训练时使用多分支结构，增强特征表示能力
    - 部署时通过重参数化转换为单分支结构，保持高效率
    - 平衡了训练时的模型容量和推理时的计算效率
    
    分支结构：
    1. Origin: 标准卷积分支
    2. Avg: 平均池化分支，捕获全局信息
    3. Prior: 先验频域分支，使用DCT初始化
    4. 1x1_kxk: 1x1和kxk组合分支
    5. 1x1: 逐点卷积分支
    6. DWS: 深度可分离卷积分支
    
    参数:
    - in_channels/out_channels: 输入输出通道数
    - kernel_size: 卷积核大小
    - stride: 步长
    - padding: 填充
    - groups: 分组数
    - dilation: 膨胀率
    - act: 激活函数
    - deploy: 是否为部署模式
    - single_init: 是否使用单一初始化
    - weight_only: 仅返回融合后的权重
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1, dilation=1,
                 act=True, internal_channels_1x1_3x3=None, deploy=False, single_init=False, weight_only=False,
                 init_hyper_para=1.0, init_hyper_gamma=1.0):
        super(OREPA, self).__init__()
        self.deploy = deploy  # 部署模式标志
        self.nonlinear = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.weight_only = weight_only  # 仅返回权重模式

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.stride = stride
        padding = autopad(kernel_size, padding, dilation)
        self.padding = padding
        self.dilation = dilation

        # ========== 部署模式 ==========
        if deploy:
            # 直接使用单个卷积层
            self.orepa_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        # ========== 训练模式 ==========
        else:
            self.branch_counter = 0

            # 分支1：原始卷积
            self.weight_orepa_origin = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), kernel_size, kernel_size))
            init.kaiming_uniform_(self.weight_orepa_origin, a=math.sqrt(0.0))
            self.branch_counter += 1

            # 分支2&3：平均池化分支（Avg + Prior）
            self.weight_orepa_avg_conv = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
            self.weight_orepa_pfir_conv = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
            init.kaiming_uniform_(self.weight_orepa_avg_conv, a=0.0)
            init.kaiming_uniform_(self.weight_orepa_pfir_conv, a=0.0)
            self.register_buffer('weight_orepa_avg_avg',
                                  torch.ones(kernel_size, kernel_size).mul(1.0 / kernel_size / kernel_size))
            self.branch_counter += 1
            self.branch_counter += 1

            # 分支4：1x1卷积
            self.weight_orepa_1x1 = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
            init.kaiming_uniform_(self.weight_orepa_1x1, a=0.0)
            self.branch_counter += 1

            # 分支5：1x1_kxk组合
            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups <= 4 else 2 * in_channels

            if internal_channels_1x1_3x3 == in_channels:
                self.weight_orepa_1x1_kxk_idconv1 = nn.Parameter(
                    torch.zeros(in_channels, int(in_channels / self.groups), 1, 1))
                # 创建单位矩阵用于恒等映射
                id_value = np.zeros((in_channels, int(in_channels / self.groups), 1, 1))
                for i in range(in_channels):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(self.weight_orepa_1x1_kxk_idconv1)
                self.register_buffer('id_tensor', id_tensor)
            else:
                self.weight_orepa_1x1_kxk_idconv1 = nn.Parameter(
                    torch.zeros(internal_channels_1x1_3x3, int(in_channels / self.groups), 1, 1))
                id_value = np.zeros((internal_channels_1x1_3x3, int(in_channels / self.groups), 1, 1))
                for i in range(internal_channels_1x1_3x3):
                    id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
                id_tensor = torch.from_numpy(id_value).type_as(self.weight_orepa_1x1_kxk_idconv1)
                self.register_buffer('id_tensor', id_tensor)

            self.weight_orepa_1x1_kxk_conv2 = nn.Parameter(
                torch.Tensor(out_channels, int(internal_channels_1x1_3x3 / self.groups), kernel_size, kernel_size))
            init.kaiming_uniform_(self.weight_orepa_1x1_kxk_conv2, a=math.sqrt(0.0))
            self.branch_counter += 1

            # 分支6：深度可分离卷积
            expand_ratio = 8
            self.weight_orepa_gconv_dw = nn.Parameter(
                torch.Tensor(in_channels * expand_ratio, 1, kernel_size, kernel_size))
            self.weight_orepa_gconv_pw = nn.Parameter(
                torch.Tensor(out_channels, int(in_channels * expand_ratio / self.groups), 1, 1))
            init.kaiming_uniform_(self.weight_orepa_gconv_dw, a=math.sqrt(0.0))
            init.kaiming_uniform_(self.weight_orepa_gconv_pw, a=math.sqrt(0.0))
            self.branch_counter += 1

            # 可学习权重向量：控制各分支的重要性
            self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
            if weight_only is False:
                self.bn = nn.BatchNorm2d(self.out_channels)

            # 频域先验初始化
            self.fre_init()

            # 初始化权重向量
            init.constant_(self.vector[0, :], 0.25 * math.sqrt(init_hyper_gamma))  # origin
            init.constant_(self.vector[1, :], 0.25 * math.sqrt(init_hyper_gamma))  # avg
            init.constant_(self.vector[2, :], 0.0 * math.sqrt(init_hyper_gamma))  # prior
            init.constant_(self.vector[3, :], 0.5 * math.sqrt(init_hyper_gamma))  # 1x1_kxk
            init.constant_(self.vector[4, :], 1.0 * math.sqrt(init_hyper_gamma))  # 1x1
            init.constant_(self.vector[5, :], 0.5 * math.sqrt(init_hyper_gamma))  # dws_conv

            # 超参数缩放
            self.weight_orepa_1x1.data = self.weight_orepa_1x1.mul(init_hyper_para)
            self.weight_orepa_origin.data = self.weight_orepa_origin.mul(init_hyper_para)
            self.weight_orepa_1x1_kxk_conv2.data = self.weight_orepa_1x1_kxk_conv2.mul(init_hyper_para)
            self.weight_orepa_avg_conv.data = self.weight_orepa_avg_conv.mul(init_hyper_para)
            self.weight_orepa_pfir_conv.data = self.weight_orepa_pfir_conv.mul(init_hyper_para)
            self.weight_orepa_gconv_dw.data = self.weight_orepa_gconv_dw.mul(math.sqrt(init_hyper_para))
            self.weight_orepa_gconv_pw.data = self.weight_orepa_gconv_pw.mul(math.sqrt(init_hyper_para))

            if single_init:
                self.single_init()

    def fre_init(self):
        """频域先验初始化"""
        prior_tensor = torch.Tensor(self.out_channels, self.kernel_size, self.kernel_size)
        half_fg = self.out_channels / 2
        for i in range(self.out_channels):
            for h in range(3):
                for w in range(3):
                    if i < half_fg:
                        prior_tensor[i, h, w] = math.cos(math.pi * (h + 0.5) * (i + 1) / 3)
                    else:
                        prior_tensor[i, h, w] = math.cos(math.pi * (w + 0.5) * (i + 1 - half_fg) / 3)
        self.register_buffer('weight_orepa_prior', prior_tensor)

    def weight_gen(self):
        """
        生成融合后的卷积权重
        
        功能：将多个分支的权重按可学习权重向量加权融合
        
        返回:
        - weight: 融合后的卷积权重 [out_channels, in_channels/groups, k, k]
        """
        # 分支1：原始卷积
        weight_orepa_origin = torch.einsum('oihw,o->oihw', self.weight_orepa_origin, self.vector[0, :])

        # 分支2：平均池化分支
        weight_orepa_avg = torch.einsum('oihw,hw->oihw', self.weight_orepa_avg_conv, self.weight_orepa_avg_avg)
        weight_orepa_avg = torch.einsum('oihw,o->oihw',
                                        torch.einsum('oi,hw->oihw', self.weight_orepa_avg_conv.squeeze(3).squeeze(2),
                                                     self.weight_orepa_avg_avg), self.vector[1, :])

        # 分支3：先验分支
        weight_orepa_pfir = torch.einsum('oihw,o->oihw',
                                         torch.einsum('oi,ohw->oihw', self.weight_orepa_pfir_conv.squeeze(3).squeeze(2),
                                                      self.weight_orepa_prior), self.vector[2, :])

        # 分支4：1x1_kxk组合
        weight_orepa_1x1_kxk_conv1 = None
        if hasattr(self, 'weight_orepa_1x1_kxk_idconv1'):
            weight_orepa_1x1_kxk_conv1 = (self.weight_orepa_1x1_kxk_idconv1 + self.id_tensor).squeeze(3).squeeze(2)
        elif hasattr(self, 'weight_orepa_1x1_kxk_conv1'):
            weight_orepa_1x1_kxk_conv1 = self.weight_orepa_1x1_kxk_conv1.squeeze(3).squeeze(2)
        else:
            raise NotImplementedError
        weight_orepa_1x1_kxk_conv2 = self.weight_orepa_1x1_kxk_conv2

        if self.groups > 1:
            g = self.groups
            t, ig = weight_orepa_1x1_kxk_conv1.size()
            o, tg, h, w = weight_orepa_1x1_kxk_conv2.size()
            weight_orepa_1x1_kxk_conv1 = weight_orepa_1x1_kxk_conv1.view(g, int(t / g), ig)
            weight_orepa_1x1_kxk_conv2 = weight_orepa_1x1_kxk_conv2.view(g, int(o / g), tg, h, w)
            weight_orepa_1x1_kxk = torch.einsum('gti,gothw->goihw', weight_orepa_1x1_kxk_conv1,
                                                weight_orepa_1x1_kxk_conv2).reshape(o, ig, h, w)
        else:
            weight_orepa_1x1_kxk = torch.einsum('ti,othw->oihw', weight_orepa_1x1_kxk_conv1, weight_orepa_1x1_kxk_conv2)
        weight_orepa_1x1_kxk = torch.einsum('oihw,o->oihw', weight_orepa_1x1_kxk, self.vector[3, :])

        # 分支5：1x1卷积
        weight_orepa_1x1 = 0
        if hasattr(self, 'weight_orepa_1x1'):
            weight_orepa_1x1 = transVI_multiscale(self.weight_orepa_1x1, self.kernel_size)
            weight_orepa_1x1 = torch.einsum('oihw,o->oihw', weight_orepa_1x1, self.vector[4, :])

        # 分支6：深度可分离卷积
        weight_orepa_gconv = self.dwsc2full(self.weight_orepa_gconv_dw, self.weight_orepa_gconv_pw,
                                            self.in_channels, self.groups)
        weight_orepa_gconv = torch.einsum('oihw,o->oihw', weight_orepa_gconv, self.vector[5, :])

        # 融合所有分支
        weight = weight_orepa_origin + weight_orepa_avg + weight_orepa_1x1 + weight_orepa_1x1_kxk + weight_orepa_pfir + weight_orepa_gconv

        return weight

    def dwsc2full(self, weight_dw, weight_pw, groups, groups_conv=1):
        """将深度可分离卷积转换为完整卷积"""
        t, ig, h, w = weight_dw.size()
        o, _, _, _ = weight_pw.size()
        tg = int(t / groups)
        i = int(ig * groups)
        ogc = int(o / groups_conv)
        groups_gc = int(groups / groups_conv)
        weight_dw = weight_dw.view(groups_conv, groups_gc, tg, ig, h, w)
        weight_pw = weight_pw.squeeze().view(ogc, groups_conv, groups_gc, tg)
        weight_dsc = torch.einsum('cgtihw,ocgt->cogihw', weight_dw, weight_pw)
        return weight_dsc.reshape(o, int(i / groups_conv), h, w)

    def forward(self, inputs=None):
        """
        前向传播
        
        部署模式：直接使用重参数化后的卷积
        训练模式：融合多分支权重后进行卷积
        """
        # 部署模式
        if hasattr(self, 'orepa_reparam'):
            return self.nonlinear(self.orepa_reparam(inputs))

        # 生成融合权重
        weight = self.weight_gen()

        # 仅返回权重模式
        if self.weight_only is True:
            return weight

        # 使用融合权重进行卷积
        out = F.conv2d(inputs, weight, bias=None, stride=self.stride,
                        padding=self.padding, dilation=self.dilation, groups=self.groups)
        return self.nonlinear(self.bn(out))

    def get_equivalent_kernel_bias(self):
        """获取等效的卷积核和偏置（融合BN）"""
        return transI_fusebn(self.weight_gen(), self.bn)

    def switch_to_deploy(self):
        """切换到部署模式：将多分支结构重参数化为单分支"""
        if hasattr(self, 'or1x1_reparam'):
            return
        # 获取融合后的权重和偏置
        kernel, bias = self.get_equivalent_kernel_bias()
        # 创建新的卷积层
        self.orepa_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                       kernel_size=self.kernel_size, stride=self.stride,
                                       padding=self.padding, dilation=self.dilation, groups=self.groups, bias=True)
        self.orepa_reparam.weight.data = kernel
        self.orepa_reparam.bias.data = bias
        # 断开所有分支参数的梯度
        for para in self.parameters():
            para.detach_()
        # 删除训练时的参数
        self.__delattr__('weight_orepa_origin')
        self.__delattr__('weight_orepa_1x1')
        self.__delattr__('weight_orepa_1x1_kxk_conv2')
        if hasattr(self, 'weight_orepa_1x1_kxk_idconv1'):
            self.__delattr__('id_tensor')
            self.__delattr__('weight_orepa_1x1_kxk_idconv1')
        elif hasattr(self, 'weight_orepa_1x1_kxk_conv1'):
            self.__delattr__('weight_orepa_1x1_kxk_conv1')
        else:
            raise NotImplementedError
        self.__delattr__('weight_orepa_avg_avg')
        self.__delattr__('weight_orepa_avg_conv')
        self.__delattr__('weight_orepa_pfir_conv')
        self.__delattr__('weight_orepa_prior')
        self.__delattr__('weight_orepa_gconv_dw')
        self.__delattr__('weight_orepa_gconv_pw')
        self.__delattr__('bn')
        self.__delattr__('vector')

    def init_gamma(self, gamma_value):
        """初始化权重向量"""
        init.constant_(self.vector, gamma_value)

    def single_init(self):
        """单一初始化：将origin分支权重设为1，其他为0"""
        self.init_gamma(0.0)
        init.constant_(self.vector[0, :], 1.0)


# ==================== 复合瓶颈结构 ====================

class OT_Bottleneck(nn.Module):
    """
    OREPA + WTConv 瓶颈结构
    
    功能：结合OREPA和小波卷积的复合瓶颈
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = OREPA(c1, c_, k[0], 1)  # OREPA卷积
        self.cv2 = WTConv2d(c_, c2, 5, 1)  # 小波卷积
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class WT_Bottleneck(nn.Module):
    """WTConv 瓶颈结构"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = WTConv2d(c_, c2, 5, 1)  # 小波卷积
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster CSP Bottleneck with 2 convolutions"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(WT_Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(WT_Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3k with customizable kernel sizes"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(WT_Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class CSOTC(C2f):
    """
    CSP瓶颈结构，支持OT_Bottleneck或C3k
    
    功能：可切换使用OREPA或C3k作为内部模块
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else OT_Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))


class C3k2_OREPA_neck(C2f):
    """使用WTConv的C3k2模块"""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else WT_Bottleneck(self.c, self.c, shortcut, g) for _ in range(n))


if __name__ == "__main__":
    # 测试代码
    image_size = (1, 64, 224, 224)
    image = torch.rand(*image_size)
    model = CSOTC(64, 64)
    out = model(image)
    print(out.size())
