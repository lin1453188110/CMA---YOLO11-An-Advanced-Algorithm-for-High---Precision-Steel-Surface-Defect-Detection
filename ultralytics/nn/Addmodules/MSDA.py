"""
MSDA.py - 多尺度空洞注意力模块
================================

主要功能：
1. DilateAttention: 空洞注意力机制，使用膨胀卷积捕获多尺度上下文
2. MultiDilatelocalAttention: 多空洞局部注意力，同时使用多个膨胀率
3. PSABlock: 位置敏感注意力块，集成注意力和前馈网络
4. C2PSA_MSDA: 级联PSA模块，用于特征增强

技术原理：
- 空洞注意力通过设置不同的膨胀率(dilation)来扩大感受野
- 多空洞机制可以同时捕获不同尺度的特征信息
- 位置敏感注意力增强空间特征的表达能力

作者: CSDN:Snu77
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导出模块列表
__all__ = ['MultiDilatelocalAttention', 'C2PSA_MSDA']


class DilateAttention(nn.Module):
    """
    空洞注意力模块 (Dilated Attention)
    
    功能说明：
    - 使用膨胀卷积(空洞卷积)实现注意力机制
    - 通过设置不同的膨胀率(dilation)来调整感受野大小
    - 在不增加参数量的情况下捕获更大的上下文信息
    
    参数说明：
    - head_dim: 每个注意力头的维度
    - qk_scale: QK缩放因子，默认为head_dim的-0.5次方
    - attn_drop: 注意力权重的Dropout比例
    - kernel_size: 局部注意力核大小
    - dilation: 膨胀率，控制采样点的间隔
    
    注意力计算流程：
    1. 对Query进行reshape和transpose，适配多头格式
    2. 对Key进行Unfold操作，配合膨胀率实现空洞采样
    3. 计算Q@K得到注意力分数，乘以scale进行缩放
    4. Softmax归一化得到注意力权重
    5. 对Value进行加权求和
    """
    
    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        """初始化空洞注意力模块"""
        super().__init__()
        self.head_dim = head_dim  # 每个注意力头的维度
        # QK缩放因子，防止点积值过大导致softmax梯度消失
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size  # 局部注意力核大小
        # Unfold操作：实现空洞卷积的采样
        # dilation参数控制采样点的间隔
        self.unfold = nn.Unfold(kernel_size, dilation, dilation * (kernel_size - 1) // 2, 1)
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力权重的Dropout

    def forward(self, q, k, v):
        """
        前向传播 - 执行空洞注意力计算
        
        参数:
        - q: Query张量，形状 [B, d, H, W]，其中d=head_dim*num_heads
        - k: Key张量，形状 [B, d, H, W]
        - v: Value张量，形状 [B, d, H, W]
        
        返回:
        - x: 注意力输出，形状 [B, H, W, d]
        """
        # B, C//3, H, W
        B, d, H, W = q.shape
        
        # ========== Query处理 ==========
        # 将Query reshape为多头格式: [B, num_heads, N, 1, head_dim]
        # 其中N=H*W是空间位置的总数
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)
        # 形状: B, h, N, 1, d
        
        # ========== Key处理 ==========
        # 使用Unfold进行空洞采样，得到局部区域的值
        # 然后reshape为多头格式: [B, num_heads, N, head_dim, k*k]
        k = self.unfold(k).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 2, 3)
        # 形状: B, h, N, d, k*k
        
        # ========== 注意力分数计算 ==========
        # Q @ K^T 得到原始注意力分数
        # 乘以scale进行缩放，防止梯度消失
        attn = (q @ k) * self.scale  # B, h, N, 1, k*k
        
        # ========== Softmax归一化 ==========
        # 在最后一个维度(空间维度)上进行softmax
        attn = attn.softmax(dim=-1)
        
        # 应用Dropout，防止过拟合
        attn = self.attn_drop(attn)
        
        # ========== Value处理 ==========
        # 同样使用Unfold进行空洞采样
        v = self.unfold(v).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3, 2)
        # 形状: B, h, N, k*k, d
        
        # ========== 加权求和 ==========
        # 注意力权重 @ Value
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        # 转换回 [B, H, W, d] 格式
        return x


class MultiDilatelocalAttention(nn.Module):
    """
    多空洞局部注意力模块 (Multi-Dilation Local Attention)
    
    功能说明：
    - 同时使用多个不同的膨胀率进行空洞注意力计算
    - 捕获不同尺度的上下文信息，增强特征表示能力
    - 使用分组策略，将通道分成多个组，每组使用不同的膨胀率
    
    参数说明：
    - dim: 输入通道数
    - num_heads: 注意力头数量，必须是num_dilation的整数倍
    - qkv_bias: QKV投影是否使用偏置
    - qk_scale: QK缩放因子
    - attn_drop: 注意力Dropout比例
    - proj_drop: 输出投影的Dropout比例
    - kernel_size: 局部注意力核大小
    - dilation: 膨胀率列表，如[1,2,3,4]表示使用4种不同的膨胀率
    
    工作流程：
    1. QKV投影：将输入分成Q、K、V三部分
    2. 通道分组：将通道分成num_dilation组
    3. 多空洞注意力：对每组使用不同的膨胀率计算注意力
    4. 重组与投影：合并各组结果并通过线性层输出
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3, 4]):
        """初始化多空洞局部注意力模块"""
        super().__init__()
        self.dim = dim  # 输入通道数
        self.num_heads = num_heads  # 注意力头数量
        head_dim = dim // num_heads  # 每个头的维度
        self.dilation = dilation  # 膨胀率列表
        self.kernel_size = kernel_size  # 核大小
        self.scale = qk_scale or head_dim ** -0.5  # QK缩放因子
        self.num_dilation = len(dilation)  # 膨胀率的数量
        
        # 断言：num_heads必须是num_dilation的整数倍
        # 这是为了确保通道可以均匀分组
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        
        # QKV投影层：1x1卷积，将通道数扩展为3倍(分别对应Q,K,V)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        
        # 创建多个空洞注意力模块，每个使用不同的膨胀率
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        
        # 输出投影层
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        前向传播 - 执行多空洞局部注意力
        
        参数:
        - x: 输入张量，形状 [B, C, H, W]
        
        返回:
        - y4: 输出张量，形状 [B, C, H, W]
        """
        B, C, H, W = x.shape
        # 保存输入副本，用于残差连接
        y = x.clone()
        
        # ========== QKV投影与分组 ==========
        # 1. QKV投影：[B, C, H, W] -> [B, 3*C, H, W]
        # 2. Reshape：[B, 3, num_dilation, C//num_dilation, H, W]
        # 3. Transpose：[num_dilation, 3, B, C//num_dilation, H, W]
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        # 形状: num_dilation, 3, B, C//num_dilation, H, W
        
        # ========== 特征图分组 ==========
        # 将通道分成num_dilation组
        # Reshape：[B, C, H, W] -> [B, num_dilation, C//num_dilation, H, W]
        # Transpose：[num_dilation, B, H, W, C//num_dilation]
        y1 = y.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2)
        # 形状: num_dilation, B, H, W, C//num_dilation
        
        # ========== 多空洞注意力计算 ==========
        # 对每个膨胀率分别计算注意力
        for i in range(self.num_dilation):
            # 从QKV中取出第i组的Q,K,V
            # qkv[i]形状: [3, B, C//num_dilation, H, W]
            # qkv[i][0], qkv[i][1], qkv[i][2]分别对应Q,K,V
            y1[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])
            # 输出形状: B, H, W, C//num_dilation
        
        # ========== 合并多空洞结果 ==========
        # Transpose: [num_dilation, B, H, W, C//num_dilation] -> [B, H, W, num_dilation, C//num_dilation]
        # Reshape: -> [B, H, W, C]
        y2 = y1.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        
        # ========== 输出投影 ==========
        # 线性投影
        y3 = self.proj(y2)
        # Dropout
        y4 = self.proj_drop(y3)
        # 转置回通道优先格式: [B, C, H, W]
        y4 = y4.permute(0, 3, 1, 2)
        
        return y4


def autopad(k, p=None, d=1):
    """
    自动填充函数
    
    功能：根据卷积核大小、膨胀率和填充模式自动计算填充量
    
    参数:
    - k: 卷积核大小，整数或元组
    - p: 填充大小，如果为None则自动计算
    - d: 膨胀率
    
    返回:
    - p: 计算得到的填充量
    """
    """Pad to 'same' shape outputs."""
    # 如果膨胀率大于1，需要相应增大卷积核大小
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    # 如果未指定填充量，使用自动填充策略：k//2
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """
    标准卷积模块
    
    功能：封装卷积+BN+激活函数的组合
    
    参数:
    - c1: 输入通道数
    - c2: 输出通道数
    - k: 卷积核大小
    - s: 步长
    - p: 填充
    - g: 分组数
    - d: 膨胀率
    - act: 激活函数，True使用SiLU
    """
    default_act = nn.SiLU()  # 默认激活函数：SiLU/Swish

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        # 2D卷积层
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # 批归一化层
        self.bn = nn.BatchNorm2d(c2)
        # 激活函数
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """应用卷积、批归一化和激活函数"""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """融合模式下前向传播（仅卷积+激活，不含BN）"""
        return self.act(self.conv(x))


class PSABlock(nn.Module):
    """
    位置敏感注意力块 (Position-Sensitive Attention Block)
    
    功能说明：
    - 集成多头注意力机制和前馈神经网络
    - 使用残差连接和可选的短路连接
    - 实现特征的自适应增强
    
    结构组成：
    1. MultiDilatelocalAttention: 多空洞局部注意力
    2. FFN: 前馈网络(两层卷积)
    3. 残差连接
    
    参数:
    - c: 通道数
    - attn_ratio: 注意力通道占比
    - num_heads: 注意力头数量
    - shortcut: 是否使用残差连接
    """
    
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """初始化PSABlock"""
        super().__init__()
        # 多空洞局部注意力模块
        self.attn = MultiDilatelocalAttention(c)
        # 前馈网络：两层层叠的卷积
        # 第一层：通道扩展到2倍
        # 第二层：通道恢复到原始大小
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        # 是否使用残差连接
        self.add = shortcut

    def forward(self, x):
        """
        前向传播
        
        参数:
        - x: 输入张量 [B, C, H, W]
        
        返回:
        - x: 输出张量 [B, C, H, W]
        """
        # 注意力分支 + 可选残差
        x = x + self.attn(x) if self.add else self.attn(x)
        # FFN分支 + 可选残差
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA_MSDA(nn.Module):
    """
    级联多尺度空洞注意力模块 (Cascaded PSA with Multi-Scale Dilated Attention)
    
    功能说明：
    - 实现类似于PSA模块的功能，但结构更易于堆叠
    - 使用通道分割策略：一部分通道直接传递，一部分通过注意力处理
    - 可堆叠多个PSABlock增强特征表示能力
    
    参数:
    - c1: 输入通道数
    - c2: 输出通道数
    - n: PSABlock的数量
    - e: 通道扩展比例
    
    工作流程：
    1. 输入通道分割为两部分：a和b
    2. a通道直接传递
    3. b通道通过n个PSABlock进行处理
    4. 合并a和b，通过1x1卷积输出
    """
    
    def __init__(self, c1, c2, n=1, e=0.5):
        """初始化C2PSA_MSDA模块"""
        super().__init__()
        # 确保输入输出通道数相等
        assert c1 == c2
        # 计算中间通道数
        self.c = int(c1 * e)
        
        # cv1: 1x1卷积，将通道从c1扩展到2*c
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        # cv2: 1x1卷积，将通道从2*c恢复到c1
        self.cv2 = Conv(2 * self.c, c1, 1)
        
        # 创建n个PSABlock
        # attn_ratio=0.5 表示注意力使用的通道比例为0.5
        # num_heads=self.c // 64 根据通道数计算注意力头数
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """
        前向传播
        
        参数:
        - x: 输入张量 [B, C, H, W]
        
        返回:
        - output: 输出张量 [B, C, H, W]
        """
        # 通过cv1扩展通道
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        # a: 直接传递的通道 [B, c, H, W]
        # b: 需要处理的通道 [B, c, H, W]
        
        # 对b进行多尺度空洞注意力处理
        b = self.m(b)
        # [B, c, H, W]
        
        # 合并a和b，然后通过cv2恢复通道数
        return self.cv2(torch.cat((a, b), 1))


if __name__ == "__main__":
    # 测试代码
    # 生成样本图像
    image_size = (1, 64, 240, 240)
    image = torch.rand(*image_size)

    # 创建模型
    mobilenet_v1 = C2PSA_MSDA(64, 64)

    # 前向传播
    out = mobilenet_v1(image)
    print(out.size())  # 输出形状应为 [1, 64, 240, 240]
