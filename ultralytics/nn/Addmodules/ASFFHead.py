"""
ASFFHead.py - 自适应空间特征融合检测头模块
=========================================

主要功能：
1. DynamicConv: 动态卷积，根据输入特征动态调整卷积权重
2. ASFFV5: 自适应空间特征融合，动态融合多尺度特征
3. Detect_ASFF: 集成ASFF的YOLO检测头

作者: CSDN:Snu77
参考论文: Generalized Focal Loss (https://ieeexplore.ieee.org/document/9792391)
"""

import copy
import torch
import torch.nn as nn
from ultralytics.utils.tal import dist2bbox, make_anchors
import math
import torch.nn.functional as F
from timm.layers import CondConv2d

# 导出模块列表
__all__ = ['Detect_ASFF']

# 导入RFA卷积模块
from ultralytics.nn.Addmodules.RFAConv import RFAConv


class DynamicConv(nn.Module):
    """
    动态卷积层 (Dynamic Convolution)
    
    核心思想：
    - 使用CondConv（条件卷积），允许每个样本拥有不同的卷积核权重
    - 通过路由网络根据输入特征动态生成专家权重
    - 增强了模型对不同输入的适应性
    
    应用场景：
    - 需要自适应特征处理的目标检测任务
    - 特征差异较大的多尺度目标检测
    """
    
    def __init__(self, in_features, out_features, kernel_size=1, stride=1, padding='', dilation=1,
                 groups=1, bias=False, num_experts=4):
        """
        初始化动态卷积层
        
        参数:
            in_features: 输入特征维度
            out_features: 输出特征维度
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充大小
            dilation: 膨胀系数
            groups: 分组数
            bias: 是否使用偏置
            num_experts: 专家数量，决定生成多少组不同的卷积核权重
        """
        super().__init__()
        self.num_experts = num_experts
        
        # 【路由网络】根据输入特征生成各专家的权重
        # 输入: 全局平均池化后的特征向量 (batch, in_features)
        # 输出: 各专家的路由权重 (batch, num_experts)
        self.routing = nn.Linear(in_features, num_experts)
        
        # 【条件卷积】使用路由权重进行动态卷积
        # CondConv允许对每个样本使用不同的卷积核
        self.cond_conv = CondConv2d(
            in_features, out_features, kernel_size, stride, padding, dilation,
            groups, bias, num_experts
        )

    def forward(self, x):
        """
        前向传播
        
        流程:
        1. 全局平均池化: 将空间信息压缩为向量
        2. 路由权重计算: 通过Sigmoid归一化得到各专家权重
        3. 条件卷积: 使用动态权重进行卷积操作
        """
        # 步骤1: 全局平均池化，将 (B,C,H,W) → (B,C)
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)
        
        # 步骤2: 路由网络计算专家权重，使用Sigmoid归一化到[0,1]
        routing_weights = torch.sigmoid(self.routing(pooled_inputs))
        
        # 步骤3: 条件卷积，根据路由权重动态融合多个专家的卷积核
        x = self.cond_conv(x, routing_weights)
        return x


def autopad(k, p=None, d=1):
    """
    自动计算填充值，使输出尺寸与输入尺寸保持一致 ('same'填充)
    
    参数:
        k: 卷积核大小
        p: 填充大小（None时自动计算）
        d: 膨胀系数
    
    返回:
        p: 计算后的填充大小
    
    原理:
        - 当d=1时，padding = k // 2 可以保持输出尺寸不变
        - 当d>1时，需要考虑膨胀导致的实际卷积核大小变化
    """
    # 如果有膨胀(d>1)，计算实际卷积核大小
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    
    # 如果p为None，自动计算填充值
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    
    return p


class Conv(nn.Module):
    """
    标准卷积模块
    
    组成结构: Conv2d → BatchNorm2d → Activation
    
    特点:
    - 默认使用SiLU激活函数
    - 支持融合推理模式（跳过BN加速推理）
    """
    # 类属性：默认激活函数为SiLU
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        初始化标准卷积层
        
        参数:
            c1: 输入通道数
            c2: 输出通道数
            k: 卷积核大小
            s: 步长
            p: 填充大小（自动计算）
            g: 分组数
            d: 膨胀系数
            act: 激活函数（True使用默认SiLU）
        """
        super().__init__()
        
        # 步骤1: 2D卷积层
        self.conv = nn.Conv2d(
            c1, c2, k, s, 
            autopad(k, p, d),  # 自动计算填充
            groups=g, 
            dilation=d, 
            bias=False  # 使用BN时不使用偏置
        )
        
        # 步骤2: 批归一化
        self.bn = nn.BatchNorm2d(c2)
        
        # 步骤3: 激活函数
        # - True: 使用默认SiLU
        # - nn.Module: 使用指定的激活函数
        # - 其他: 使用Identity
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        前向传播: 卷积 → 归一化 → 激活
        
        参数:
            x: 输入张量 (B, C, H, W)
        
        返回:
            输出张量 (B, C', H', W')
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        融合后的前向传播（用于推理加速）
        
        说明:
        - 跳过BatchNorm，直接进行卷积和激活
        - 适用于模型导出后的推理阶段
        - 通过将BN参数融入卷积来保持等效性
        """
        return self.act(self.conv(x))


class DFL(nn.Module):
    """
    分布焦点损失(DFL)积分模块
    
    论文参考: Generalized Focal Loss
    论文链接: https://ieeexplore.ieee.org/document/9792391
    
    核心功能:
    - 将离散的回归分布转换为连续坐标值
    - 用于边界框的精确回归
    - 解决了边界框表示的不确定性问题
    
    工作原理:
    - 将预测的4个坐标值视为离散概率分布
    - 使用Softmax归一化后计算期望值
    - 期望值即为最终的坐标预测
    """

    def __init__(self, c1=16):
        """
        初始化DFL模块
        
        参数:
            c1: 离散分布的类别数，默认16
                 即每个坐标被分成16个离散区间预测
        """
        super().__init__()
        # 1x1卷积层，权重固定为[0,1,2,...,15]
        # 用于加权求和计算期望值
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        
        # 创建固定权重: [0, 1, 2, ..., c1-1]
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        
        self.c1 = c1

    def forward(self, x):
        """
        前向传播，计算离散分布的期望值
        
        参数:
            x: 输入张量，形状 (batch, channels=4*c1, anchors)
               其中4表示x,y,w,h四个坐标，c1是每个坐标的离散类别数
        
        返回:
            输出张量，形状 (batch, 4, anchors)
               四个坐标的期望值
        """
        b, c, a = x.shape  # batch, channels, anchors
        
        # 步骤1: 重排列 - 将 (b, 4*c1, a) → (b, 4, c1, a)
        # 步骤2: 维度转换 - (b, 4, c1, a) → (b, c1, 4, a)
        # 步骤3: Softmax归一化 - 在c1维度上归一化为概率分布
        # 步骤4: 卷积压缩 - (b, c1, 4, a) → (b, 1, 4, a) → (b, 4, a)
        #           这个过程相当于计算加权和，即期望值
        return self.conv(
            x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)
        ).view(b, 4, a)


class ASFFV5(nn.Module):
    """
    自适应空间特征融合模块 (Adaptive Spatial Feature Fusion)
    
    核心思想:
    - 动态学习不同尺度特征图之间的融合权重
    - 根据输入内容自适应决定如何融合多尺度信息
    - 相比简单的特征拼接，具有更强的自适应能力
    
    特点:
    - 轻量级：仅使用1x1和3x3卷积学习融合权重
    - 端到端：权重可学习，与网络共同优化
    - 即插即用：可嵌入到任何多尺度网络中
    
    三个层级的作用:
    - Level 0: 大特征图，负责检测小目标
    - Level 1: 中特征图，负责检测中目标  
    - Level 2: 小特征图，负责检测大目标
    """
    
    def __init__(self, level, ch, multiplier=1, rfb=False, vis=False, act_cfg=True):
        """
        初始化ASFFV5模块
        
        参数:
            level: 当前层的级别 (0/1/2)
                   0: 大特征图层（小目标）
                   1: 中特征图层（中目标）
                   2: 小特征图层（大目标）
            ch: 三个层的通道数元组 (ch_0, ch_1, ch_2)
            multiplier: 通道数缩放因子
            rfb: 是否使用感受野模块（减少通道数节省内存）
            vis: 是否返回可视化信息
        """
        super(ASFFV5, self).__init__()
        self.level = level  # 当前层级别
        self.vis = vis
        
        # 根据multiplier调整通道数
        # dim[0]对应小特征图(ch_0)，dim[1]对应中特征图(ch_1)，dim[2]对应大特征图(ch_2)
        self.dim = [
            int(ch[2] * multiplier),  # 大特征图通道数 (Level 0)
            int(ch[1] * multiplier),  # 中特征图通道数 (Level 1)
            int(ch[0] * multiplier)   # 小特征图通道数 (Level 2)
        ]
        
        # 当前层的目标通道数
        self.inter_dim = self.dim[self.level]
        
        # ==========================================
        # 尺度对齐层：根据当前层级对齐不同尺度的特征图
        # ==========================================
        
        if level == 0:
            # Level 0: 大特征图层，需要将中、小特征图下采样到相同尺寸
            
            # 中特征图 → 大特征图: 3x3卷积，步长2（下采样）
            self.stride_level_1 = Conv(int(ch[1] * multiplier), self.inter_dim, 3, 2)
            
            # 小特征图 → 大特征图: 先最大池化下采样，再3x3卷积
            self.stride_level_2 = Conv(int(ch[0] * multiplier), self.inter_dim, 3, 2)
            
            # 扩展回原始通道数: inter_dim → ch[2]
            self.expand = Conv(self.inter_dim, int(ch[2] * multiplier), 3, 1)
            
        elif level == 1:
            # Level 1: 中特征图层
            
            # 大特征图 → 中特征图: 1x1卷积压缩通道
            self.compress_level_0 = Conv(int(ch[2] * multiplier), self.inter_dim, 1, 1)
            
            # 小特征图 → 中特征图: 3x3卷积，步长2（下采样）
            self.stride_level_2 = Conv(int(ch[0] * multiplier), self.inter_dim, 3, 2)
            
            # 扩展回原始通道数: inter_dim → ch[1]
            self.expand = Conv(self.inter_dim, int(ch[1] * multiplier), 3, 1)
            
        elif level == 2:
            # Level 2: 小特征图层，需要将大、中特征图上采样到相同尺寸
            
            # 大特征图 → 小特征图: 1x1卷积 + 双线性插值上采样4倍
            self.compress_level_0 = Conv(int(ch[2] * multiplier), self.inter_dim, 1, 1)
            
            # 中特征图 → 小特征图: 1x1卷积 + 双线性插值上采样2倍
            self.compress_level_1 = Conv(int(ch[1] * multiplier), self.inter_dim, 1, 1)
            
            # 扩展回原始通道数: inter_dim → ch[0]
            self.expand = Conv(self.inter_dim, int(ch[0] * multiplier), 3, 1)

        # ==========================================
        # 权重学习层：学习三个尺度特征的融合权重
        # ==========================================
        
        # RFB (Receptive Field Block) 模式使用更少的通道节省内存
        compress_c = 8 if rfb else 16
        
        # 三个尺度特征各自的权重学习分支
        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)  # Level 0权重
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)  # Level 1权重
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)  # Level 2权重
        
        # 融合三个尺度的权重: compress_c*3 → 3
        # 输出的3个通道分别对应Level 0, 1, 2的融合权重
        self.weight_levels = Conv(compress_c * 3, 3, 1, 1)

    def forward(self, x):
        """
        前向传播：多尺度特征自适应融合
        
        参数:
            x: 三个尺度特征图的列表 [小, 中, 大]
               x[0]: 小特征图 (高分辨率，低层语义)
               x[1]: 中特征图 (中等分辨率，中层语义)
               x[2]: 大特征图 (低分辨率，高层语义)
        
        返回:
            融合后的特征图，尺寸与对应层级的特征图相同
        """
        # 获取三个尺度的特征图
        x_level_0 = x[2]  # 大特征图 (Low Resolution)
        x_level_1 = x[1]  # 中特征图 (Medium Resolution)
        x_level_2 = x[0]  # 小特征图 (High Resolution)
        
        # ==========================================
        # 步骤1: 尺度对齐 - 将三个特征图调整为相同尺寸
        # ==========================================
        
        if self.level == 0:
            # Level 0: 保持大特征图不变，将中、小特征图下采样
            
            level_0_resized = x_level_0  # 保持不变
            
            # 中→大: 直接3x3卷积下采样
            level_1_resized = self.stride_level_1(x_level_1)
            
            # 小→大: 先最大池化下采样，再卷积
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
            
        elif self.level == 1:
            # Level 1: 需要上采样大特征图，下采样小特征图
            
            # 大→中: 1x1卷积压缩通道，然后上采样2倍
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            
            level_1_resized = x_level_1  # 保持不变
            
            # 小→中: 3x3卷积下采样
            level_2_resized = self.stride_level_2(x_level_2)
            
        elif self.level == 2:
            # Level 2: 将大、中特征图上采样
            
            # 大→小: 1x1卷积压缩通道，然后上采样4倍
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            
            # 中→小: 1x1卷积压缩通道，然后上采样2倍
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(x_level_1_compressed, scale_factor=2, mode='nearest')
            
            level_2_resized = x_level_2  # 保持不变
        
        # ==========================================
        # 步骤2: 学习融合权重
        # ==========================================
        
        # 三个分支分别学习各自的权重
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        
        # 拼接三个权重组: (B, 3*compress_c, H, W)
        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1
        )
        
        # 融合权重: (B, 3*compress_c, H, W) → (B, 3, H, W)
        levels_weight = self.weight_levels(levels_weight_v)
        
        # Softmax归一化，确保三个权重和为1
        levels_weight = F.softmax(levels_weight, dim=1)
        
        # ==========================================
        # 步骤3: 加权融合
        # ==========================================
        
        # 三个尺度特征按学习到的权重加权求和
        fused_out_reduced = (
            level_0_resized * levels_weight[:, 0:1, :, :] +  # Level 0 * 权重0
            level_1_resized * levels_weight[:, 1:2, :, :] +  # Level 1 * 权重1
            level_2_resized * levels_weight[:, 2:, :, :]    # Level 2 * 权重2
        )
        
        # ==========================================
        # 步骤4: 通道调整
        # ==========================================
        
        # 将通道数从inter_dim扩展回对应层级的原始通道数
        out = self.expand(fused_out_reduced)
        
        # 如果需要可视化，返回中间结果
        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


class DWConv(Conv):
    """
    深度可分离卷积 (Depth-wise Convolution)
    
    原理:
    - 每个输入通道使用独立的卷积核进行处理
    - 大幅减少参数量和计算量
    - 相比标准卷积，轻量化约8-9倍
    
    计算: 参数量 = k*k*C (相比标准卷积的 k*k*C*C 大幅减少)
    """
    
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        初始化深度可分离卷积
        
        参数:
            c1: 输入通道数
            c2: 输出通道数
            k: 卷积核大小
            s: 步长
            d: 膨胀系数
            act: 激活函数
        """
        # 深度可分离卷积的分组数 = 输入通道数 = 输出通道数（分组=通道数时为深度卷积）
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class Detect_ASFF(nn.Module):
    """
    集成ASFF的YOLOv8检测头
    
    改进点:
    - 将ASFF多尺度特征融合集成到检测头中
    - 使用DynamicConv增强特征提取能力
    - 支持端到端训练和推理
    
    特点:
    - 动态融合：自适应学习多尺度特征权重
    - 双分支设计：回归分支(cv2)和分类分支(cv3)
    - 支持DDF(分布焦点损失)和NMS后处理
    """
    
    # 类属性：用于配置模型行为
    dynamic = False      # 是否强制重建网格
    export = False       # 是否为导出模式
    end2end = False      # 是否为端到端模式
    max_det = 300        # 最大检测数量
    shape = None         # 特征图形状缓存
    anchors = torch.empty(0)  # 锚框
    strides = torch.empty(0)  # 步长

    def __init__(self, nc=80, ch=(), multiplier=1, rfb=False):
        """
        初始化检测头
        
        参数:
            nc: 类别数量
            ch: 三个检测层的通道数元组 (P3, P4, P5)
            multiplier: ASFF融合的通道缩放因子
            rfb: 是否使用感受野模块
        """
        super().__init__()
        
        # 类别数量
        self.nc = nc
        # 检测层数量（通常为3，对应多尺度检测）
        self.nl = len(ch)
        # DFL的通道数（用于分布焦点损失）
        self.reg_max = 16
        # 每个anchor的输出数量 = 4*reg_max(回归) + nc(类别)
        self.no = nc + self.reg_max * 4
        
        # 初始化步长（后续根据特征图计算）
        self.stride = torch.zeros(self.nl)
        
        # 计算回归分支和分类分支的中间通道数
        # reg_max*4确保有足够的通道用于DFL
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        # 分类分支通道数，限制在100以内
        c3 = max(ch[0], min(self.nc, 100))
        
        # ==========================================
        # 回归分支 (cv2): 预测边界框坐标
        # 结构: DynamicConv → Conv → 1x1Conv → 4*reg_max通道
        # ==========================================
        self.cv2 = nn.ModuleList([
            nn.Sequential(
                DynamicConv(x, c2, 3),  # 动态卷积增强特征
                Conv(c2, c2, 3),        # 标准卷积
                nn.Conv2d(c2, 4 * self.reg_max, 1)  # 输出4*reg_max通道
            )
            for x in ch
        ])
        
        # ==========================================
        # 分类分支 (cv3): 预测类别概率
        # 结构: DWConv+Conv → DWConv+Conv → 1x1Conv → nc通道
        # ==========================================
        self.cv3 = nn.ModuleList([
            nn.Sequential(
                # 第一个DWConv块
                nn.Sequential(
                    DWConv(x, x, 3),  # 深度可分离卷积
                    Conv(x, c3, 1)     # 1x1卷积调整通道
                ),
                # 第二个DWConv块
                nn.Sequential(
                    DWConv(c3, c3, 3),  # 深度可分离卷积
                    Conv(c3, c3, 1)      # 1x1卷积调整通道
                ),
                # 最终输出nc个类别的概率
                nn.Conv2d(c3, self.nc, 1)
            )
            for x in ch
        ])
        
        # ==========================================
        # DFL模块: 分布焦点损失
        # ==========================================
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        
        # ==========================================
        # ASFF多尺度特征融合
        # ==========================================
        # 三个层级的自适应特征融合模块
        self.l0_fusion = ASFFV5(level=0, ch=ch, multiplier=multiplier, rfb=rfb)  # 大目标层融合
        self.l1_fusion = ASFFV5(level=1, ch=ch, multiplier=multiplier, rfb=rfb)  # 中目标层融合
        self.l2_fusion = ASFFV5(level=2, ch=ch, multiplier=multiplier, rfb=rfb)  # 小目标层融合
        
        # 端到端模式下需要的额外分支
        if self.end2end:
            # one2one分支用于直接的端到端检测
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 三个尺度特征图的列表 [P3, P4, P5]
               P3: 小特征图（高分辨率，检测小目标）
               P4: 中特征图（中等分辨率，检测中目标）
               P5: 大特征图（低分辨率，检测大目标）
        
        返回:
            训练模式: 三个尺度融合后的原始输出
            推理模式: 解码后的检测结果 (边界框+类别概率)
        """
        # ==========================================
        # 步骤1: ASFF多尺度特征融合
        # ==========================================
        x1 = self.l0_fusion(x)  # Level 0特征融合
        x2 = self.l1_fusion(x)  # Level 1特征融合
        x3 = self.l2_fusion(x)  # Level 2特征融合
        
        # 重新排列为 [小目标, 中目标, 大目标]
        x = [x3, x2, x1]
        
        # 端到端模式
        if self.end2end:
            return self.forward_end2end(x)
        
        # ==========================================
        # 步骤2: 任务专属卷积
        # ==========================================
        # 对每个尺度的融合特征分别应用回归和分类卷积
        for i in range(self.nl):
            # 拼接回归分支和分类分支的输出
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        
        # 训练模式返回原始输出
        if self.training:
            return x
        
        # ==========================================
        # 步骤3: 推理后处理
        # ==========================================
        y = self._inference(x)
        
        # 根据导出模式决定返回值
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        端到端模式的前向传播
        
        特点:
        - 同时输出one2many和one2one结果
        - one2many用于知识蒸馏
        - one2one用于最终检测
        """
        # 分离梯度以避免互相影响
        x_detach = [xi.detach() for xi in x]
        
        # one2one分支的预测
        one2one = [
            torch.cat((
                self.one2one_cv2[i](x_detach[i]),  # 回归分支
                self.one2one_cv3[i](x_detach[i])   # 分类分支
            ), 1)
            for i in range(self.nl)
        ]
        
        # one2many分支的预测
        for i in range(self.nl):
            x[i] = torch.cat((
                self.cv2[i](x[i]),  # 回归分支
                self.cv3[i](x[i])   # 分类分支
            ), 1)
        
        # 训练模式返回两个分支的输出
        if self.training:
            return {"one2many": x, "one2one": one2one}
        
        # 使用one2one结果进行最终推理
        y = self._inference(one2one)
        
        # NMS后处理
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """
        推理时的解码过程
        
        步骤:
        1. 拼接所有检测层的输出
        2. 生成锚框
        3. 分割边界框和类别预测
        4. DFL解码边界框
        5. 类别概率Sigmoid归一化
        """
        shape = x[0].shape  # BCHW
        
        # 步骤1: 拼接所有检测层输出
        # 将每个检测层的输出从 (B, C, H*W) 拼接为 (B, C_total, 1)
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        
        # 步骤2: 生成锚框和步长
        if self.dynamic or self.shape != shape:
            # 根据特征图生成锚框中心点和步长
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape
        
        # 步骤3: 分割边界框和类别预测
        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:
            # TensorFlow格式的特殊处理
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            # 标准分割: 前4*reg_max通道是边界框，后面是类别概率
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        
        # 步骤4: 特殊格式的边界框解码
        if self.export and self.format in {"tflite", "edgetpu"}:
            # TFLite/EdgeTPU需要预计算归一化因子提高数值稳定性
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor(
                [grid_w, grid_h, grid_w, grid_h], device=box.device
            ).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(
                self.dfl(box) * norm,
                self.anchors.unsqueeze(0) * norm[:, :2]
            )
        else:
            # 标准DFL解码: 将离散分布转换为连续坐标
            dbox = self.decode_bboxes(
                self.dfl(box),  # DFL解码
                self.anchors.unsqueeze(0)  # 锚框
            ) * self.strides  # 乘以步长得到实际坐标
        
        # 步骤5: 类别概率Sigmoid归一化
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """
        初始化检测头的偏置
        
        说明:
        - 边界框分支偏置设为1（较高的初始值）
        - 类别分支偏置根据COCO数据集的先验设置
        """
        m = self
        
        # 遍历所有检测层
        for a, b, s in zip(m.cv2, m.cv3, self.stride):
            # 边界框分支: 偏置设为1（初始覆盖范围较大）
            a[-1].bias.data[:] = 1.0
            
            # 类别分支: 根据尺度和类别数计算偏置
            # 公式来源: 基于先验概率5/640和特征图尺寸
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)
        
        # 端到端模式的one2one分支初始化
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, self.stride):
                a[-1].bias.data[:] = 1.0
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)

    def decode_bboxes(self, bboxes, anchors):
        """
        解码边界框坐标
        
        参数:
            bboxes: 预测的边界框偏移量
            anchors: 锚框
        
        返回:
            解码后的边界框坐标
        """
        return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

    @staticmethod
    def postprocess(preds, max_det, nc=80):
        """
        后处理：NMS和Top-K选择
        
        参数:
            preds: 原始预测 (batch_size, num_anchors, 4+nc)
            max_det: 每张图最大检测数量
            nc: 类别数量
        
        返回:
            处理后的检测结果 (batch_size, min(max_det, anchors), 6)
            格式: [x, y, w, h, confidence, class_id]
        """
        batch_size, anchors, _ = preds.shape
        
        # 分割边界框和类别分数
        boxes, scores = preds.split([4, nc], dim=-1)
        
        # Top-K选择最高分数的检测框
        # 找出每个位置最高类别的分数和索引
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        
        # 根据索引收集对应的边界框和分数
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        
        # 展平分数并再次Top-K选择
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        
        # 生成批次索引
        i = torch.arange(batch_size)[..., None]
        
        # 组装最终结果
        return torch.cat([
            boxes[i, index // nc],      # 边界框坐标
            scores[..., None],           # 置信度
            (index % nc)[..., None].float()  # 类别索引
        ], dim=-1)


if __name__ == "__main__":
    # 测试代码：验证模块功能
    
    # 创建三个尺度的测试输入
    # 小特征图: 1x64x32x32
    image1 = (1, 64, 32, 32)
    # 中特征图: 1x128x16x16
    image2 = (1, 128, 16, 16)
    # 大特征图: 1x256x8x8
    image3 = (1, 256, 8, 8)
    
    # 生成随机测试数据
    image1 = torch.rand(image1)
    image2 = torch.rand(image2)
    image3 = torch.rand(image3)
    image = [image1, image2, image3]
    
    # 通道数配置
    channel = (64, 128, 256)
    
    # 创建检测头模型
    mobilenet_v1 = Detect_ASFF(nc=80, ch=channel)
    
    # 前向传播测试
    out = mobilenet_v1(image)
    print("输出形状:", out[0].shape if isinstance(out, tuple) else out.shape)
