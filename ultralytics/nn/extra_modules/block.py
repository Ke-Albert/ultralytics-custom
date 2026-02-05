from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..modules import Conv, RepConv
from .attention import EMA, PatchEmbed, AgentAttention

__all__ = (
    "CED",
    "GatedFFN",
    "AgentBlock",
    "C2fFE",
)


class GatedFFN(nn.Module):
    """
    RemDet网络中GatedFFN实现
    门控前馈网络（Gated Feed-Forward Network）
    c1	int	无默认值	输入特征图的通道数（必须指定，如 backbone 输出的 256 通道）
    c2	int	无默认值	输出特征图的通道数（需与后续模块输入通道匹配，如 neck 所需的 512 通道）
    n	int	1	深度可分离卷积的堆叠次数（控制模块深度，n 越大特征提取越充分）
    shortcut	bool	False	是否启用残差连接（缓解梯度消失，需满足 c1 == c2 才生效）
    e	int	3	通道扩展系数（控制中间特征通道数，self.c = c2 * e，平衡表达与效率）

    1. 动态门控机制：精准筛选特征
    传统 FFN 对所有特征通道一视同仁，冗余特征会干扰后续检测；
    GatedFFN 用 z 生成动态权重，通过 x * GELU(z) 抑制无用特征（如背景噪声），聚焦目标相关特征（如边缘、纹理），尤其适合小目标 / 遮挡目标检测。
    2. 重参数化 + 深度可分离卷积：高效轻量
    RepDWConv：训练时多分支保证表达能力，推理时融合为单分支，速度提升 30%+；
    深度可分离卷积：参数量仅为普通卷积的 1/groups（此处 groups=self.c，参数量降至 1/self.c），计算量大幅降低，适合实时场景（如端侧设备）。
    3. 通道扩展 - 压缩：兼顾表达与冗余
    先通过 self.proj 扩展通道（2*self.c），让中间特征能捕捉更丰富的信息；
    再通过 self.cv2 压缩回 c2，减少冗余参数，避免过拟合。
    4. 可配置化设计：适配不同任务
    可通过 n 调整特征提取深度（小目标检测用 n=2~3，大目标用 n=1）；
    可通过 e 调整通道扩展幅度（复杂场景用 e=4，简单场景用 e=2）；
    可通过 shortcut 控制残差连接（深层模型启用，浅层模型禁用）。
    """

    def __init__(self, c1, c2, num_blocks, shortcut=False, expansion=3):
        super().__init__()
        self.n = num_blocks
        self.c = int(c2 * expansion)
        self.prj = Conv(c1, 2 * self.c, k=1, s=1)
        self.rep = RepConv(self.c, self.c)
        self.m = nn.ModuleList(
            Conv(self.c, self.c, k=3, s=1, g=self.c) for _ in range(self.n - 1))
        self.act = nn.GELU()
        self.cv2 = Conv(self.c, c2, k=1, s=1, act=False)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        shortcut = x.clone()
        x, z = self.prj(x).split([self.c, self.c], 1)
        x = self.rep(x)
        if self.n != 1:
            for m in self.m:
                x = m(x)
        x = x * self.act(z)
        x = self.cv2(x)
        return x + shortcut if self.add else x


class CED(nn.Module):
    """
    RemDet中CED模块的实现
    """

    def __init__(self, c1, c2, expansion=0.5):
        super().__init__()
        self.c = int(c2 * expansion)
        self.cv1 = Conv(c1, self.c, k=1, s=1)
        self.cv2 = Conv(self.c * 4, c2, k=1, s=1)
        self.dwconv = Conv(self.c, self.c, k=3, s=1, g=self.c)

    def forward(self, x):
        x = self.dwconv(self.cv1(x))
        x = torch.cat([
            x[..., ::2, ::2], x[..., 1::2, ::2],
            x[..., ::2, 1::2], x[..., 1::2, 1::2],
        ], dim=1)
        x = self.cv2(x)
        return x


class C2fFE(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, if_backbone=False, e: float = 0.5):
        """
        Initialize an improvement C2f block.
        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of FEBlock blocks.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(FEBlock(self.c, if_backbone) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class FEBlock(nn.Module):
    def __init__(self, c_in, if_backbone=False):
        super().__init__()
        self.c_in = c_in
        self.pconv = Pconv(c_in, 4)
        self.m = nn.ModuleList([Conv(c_in, c_in, 1, 1)])
        if if_backbone:
            self.m.append(Conv(c_in, c_in, 1, 1))
        else:
            self.m.append(EMA(c_in, 8))

    def forward(self, x):
        y = self.pconv(x)
        for m in self.m:
            y = m(y)
        return x + y


class Pconv(nn.Module):
    def __init__(self, c_in, cp_factor):
        super().__init__()
        self.c_in = c_in
        self.cp_factor = cp_factor
        self.conv = nn.Conv2d(c_in // cp_factor, c_in // cp_factor, kernel_size=3, padding=1)
        assert c_in // cp_factor > 0

    def forward(self, x):
        x1, x2 = torch.split(x, [self.c_in // self.cp_factor, (self.c_in // self.cp_factor) * (self.cp_factor - 1)],
                             dim=1)
        x1 = self.conv(x1)
        return torch.cat([x1, x2], dim=1)


class AgentBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.embed_dim = 16 * c_in
        self.patch_embed = PatchEmbed(patch_size=4, in_chans=c_in, embed_dim=self.embed_dim)
        self.agent_attn = AgentAttention(dim=self.embed_dim, num_patches=25)

    def forward(self, x):
        B, C, H, W = x.shape
        assert self.c_in == C
        patch_size = 4

        x, (h, w) = self.patch_embed(x)  # h=H//p w=W//p
        x = self.agent_attn(x, h, w)  # [B,H*W/p**2,p**2*C]
        x_feat = x.permute(0, 2, 1).reshape(B, self.embed_dim, h, w)
        x_feat = x_feat.view(B, self.c_in, patch_size, patch_size, h, w)
        x_feat = x_feat.permute(0, 1, 4, 2, 5, 3).contiguous().view(B, self.c_in, h * patch_size, w * patch_size)

        return x_feat
