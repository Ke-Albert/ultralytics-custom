from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_
from torch import Tensor, nn

__all__ = (
    "EMA",
    "AgentAttention",
    "BiLevelRoutingAttention",
)


class EMA(nn.Module):
    """https://doi.org/10.1038/s41598-025-86239-w EMA(Efficient multi-scal Attention) implementation.

    https://arxiv.org/abs/2305.13563
    """

    def __init__(self, c_in, groups):
        super().__init__()
        self.groups = groups
        assert c_in // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(c_in // self.groups, c_in // self.groups)
        self.conv1x1 = nn.Conv2d(c_in // self.groups, c_in // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(c_in // self.groups, c_in // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=20, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        self.patch_size = [patch_size, patch_size]
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

        self.img_size = [img_size, img_size]
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, self.grid_size


class AgentAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_patches,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        agent_num=49,
        **kwargs,
    ):
        super().__init__()
        # 固定参数：初始化时确定，不随输入变化
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qkv_scale = qk_scale
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        self.agent_num = agent_num
        self._layers_initialized = False

        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        trunc_normal_(self.an_bias, std=0.02)
        trunc_normal_(self.na_bias, std=0.02)
        pool_size = int(agent_num**0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)

        self.q = None
        self.kv = None
        self.proj = None
        self.sr = None
        self.norm = None
        self.dwc = None
        self.ah_bias = None
        self.aw_bias = None
        self.ha_bias = None
        self.wa_bias = None

        self.dim = None
        self.num_patches = None
        self.window_size = None
        self.head_dim = None
        self.scale = None

        assert dim % self.num_heads == 0, f"dim {dim} should be divided by num_heads {self.num_heads}"
        self.dim = dim
        self.num_patches = num_patches
        self.window_size = (int(num_patches**0.5), int(num_patches**0.5))
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=self.qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=self.qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=self.qkv_bias)

        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.ah_bias = nn.Parameter(
            torch.zeros(1, self.num_heads, self.agent_num, self.window_size[0] // self.sr_ratio, 1)
        )
        self.aw_bias = nn.Parameter(
            torch.zeros(1, self.num_heads, self.agent_num, 1, self.window_size[1] // self.sr_ratio)
        )
        self.ha_bias = nn.Parameter(torch.zeros(1, self.num_heads, self.window_size[0], 1, self.agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, self.num_heads, 1, self.window_size[1], self.agent_num))
        trunc_normal_(self.ah_bias, std=0.02)
        trunc_normal_(self.aw_bias, std=0.02)
        trunc_normal_(self.ha_bias, std=0.02)
        trunc_normal_(self.wa_bias, std=0.02)

    def forward(self, x, H, W):
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = self.head_dim
        q = self.q(x)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        agent_tokens = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n // self.sr_ratio**2, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n // self.sr_ratio**2, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        kv_size = (self.window_size[0] // self.sr_ratio, self.window_size[1] // self.sr_ratio)
        position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode="bilinear")
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode="bilinear")
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, H // self.sr_ratio, W // self.sr_ratio, c).permute(0, 3, 1, 2)
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v, size=(H, W), mode="bilinear")
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)  # [B,h*w/p**2,p**2*c]

        return x


class TopkRouting(nn.Module):
    """differentiable topk routing with scaling.

    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights.
    """

    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim**-0.5
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor

        Returns:
            r_weight, topk_index: (n, p^2, topk) tensor.
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)  # per-window pooling -> (n, p^2, c)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)  # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)  # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit)  # (n, p^2, k)

        return r_weight, topk_index


class KVGather(nn.Module):
    def __init__(self, mul_weight="none"):
        super().__init__()
        assert mul_weight in ["none", "soft", "hard"]
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        """r_idx: (n, p^2, topk) tensor r_weight: (n, p^2, topk) tensor kv: (n, p^2, w^2, c_kq+c_v).

        Returns:
            (n, p^2, topk, w^2, c_kq+c_v) tensor.
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel?
        topk_kv = torch.gather(
            kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
            # (n, p^2, p^2, w^2, c_kv) without mem cpy
            dim=2,
            index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv),
            # (n, p^2, k, w^2, c_kv)
        )

        if self.mul_weight == "soft":
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv  # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == "hard":
            raise NotImplementedError("differentiable hard routing TBA")
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv


class BiLevelRoutingAttention(nn.Module):
    """n_win: number of windows in one side (so the actual number of windows is n_win*n_win) kv_per_win: for
    kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is
    kv_per_win*kv_per_win. topk: topk for window filtering param_attention: 'qkvo'-linear for q,k,v and o, 'none':
    param free attention param_routing: extra linear for routing diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        n_win=8,
        qk_dim=None,
        qk_scale=None,
        kv_per_win=4,
        kv_downsample_ratio=4,
        kv_downsample_kernel=None,
        kv_downsample_mode="identity",
        topk=4,
        param_attention="qkvo",
        param_routing=False,
        diff_routing=False,
        soft_routing=False,
        side_dwconv=3,
        auto_pad=True,
    ):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, (
            "qk_dim and dim must be divisible by num_heads!"
        )
        self.scale = qk_scale or self.qk_dim**-0.5

        ################side_dwconv (i.e. LCE in ShuntedTransformer)###########
        self.lepe = (
            nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2, groups=dim)
            if side_dwconv > 0
            else lambda x: torch.zeros_like(x)
        )

        ################ global routing setting #################
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # router
        assert not (self.param_routing and not self.diff_routing)  # cannot be with_param=True and diff_routing=False
        self.router = TopkRouting(
            qk_dim=self.qk_dim,
            qk_scale=self.scale,
            topk=self.topk,
            diff_routing=self.diff_routing,
            param_routing=self.param_routing,
        )
        if self.soft_routing:  # soft routing, always diffrentiable (if no detach)
            mul_weight = "soft"
        elif self.diff_routing:  # hard differentiable routing
            mul_weight = "hard"
        else:  # hard non-differentiable routing
            mul_weight = "none"
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == "qkvo":
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == "qkv":
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f"param_attention mode {self.param_attention} is not supported!")

        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == "ada_avgpool":
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == "ada_maxpool":
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == "maxpool":
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == "avgpool":
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == "identity":  # no kv downsampling
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == "fracpool":
            # assert self.kv_downsample_ratio is not None
            # assert self.kv_downsample_kenel is not None
            # TODO: fracpool
            # 1. kernel size should be input size dependent
            # 2. there is a random factor, need to avoid independent sampling for k and v
            raise NotImplementedError("fracpool policy is not implemented yet!")
        elif kv_downsample_mode == "conv":
            # TODO: need to consider the case where k != v so that need two downsample modules
            raise NotImplementedError("conv policy is not implemented yet!")
        else:
            raise ValueError(f"kv_down_sample_mode {self.kv_downsaple_mode} is not supported!")

        # softmax for local attention
        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad = auto_pad

    def forward(self, x, ret_attn_mask=False):
        """x: NHWC tensor.

        Returns:
            NHWC tensor.
        """
        # NOTE: use padding for semantic segmentation
        ###################################################
        x = x.permute(0, 2, 3, 1).contiguous()

        if self.auto_pad:
            N, H_in, W_in, C = x.size()

            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(
                x,
                (
                    0,
                    0,  # dim=-1
                    pad_l,
                    pad_r,  # dim=-2
                    pad_t,
                    pad_b,
                ),
            )  # dim=-3
            _, H, W, _ = x.size()  # padded size
        else:
            _N, H, W, _C = x.size()
            assert H % self.n_win == 0 and W % self.n_win == 0
        ###################################################

        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        #################qkv projection###################
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        # NOTE: separate kv if there were memory leak issue caused by gather
        q, kv = self.qkv(x)

        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        q_pix = rearrange(q, "n p2 h w c -> n p2 (h w) c")
        kv_pix = self.kv_down(rearrange(kv, "n p2 h w c -> (n p2) c h w"))
        kv_pix = rearrange(kv_pix, "(n j i) c h w -> n (j i) (h w) c", j=self.n_win, i=self.n_win)

        q_win, k_win = (
            q.mean([2, 3]),
            kv[..., 0 : self.qk_dim].mean([2, 3]),
        )  # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)

        ##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        lepe = self.lepe(
            rearrange(
                kv[..., self.qk_dim :], "n (j i) h w c -> n c (j h) (i w)", j=self.n_win, i=self.n_win
            ).contiguous()
        )
        lepe = rearrange(lepe, "n c (j h) (i w) -> n (j h) (i w) c", j=self.n_win, i=self.n_win)

        ############ gather q dependent k/v #################

        r_weight, r_idx = self.router(q_win, k_win)  # both are (n, p^2, topk) tensors

        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)  # (n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
        # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)

        ######### do attention as normal ####################
        k_pix_sel = rearrange(
            k_pix_sel, "n p2 k w2 (m c) -> (n p2) m c (k w2)", m=self.num_heads
        )  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        v_pix_sel = rearrange(
            v_pix_sel, "n p2 k w2 (m c) -> (n p2) m (k w2) c", m=self.num_heads
        )  # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q_pix = rearrange(
            q_pix, "n p2 w2 (m c) -> (n p2) m w2 c", m=self.num_heads
        )  # to BMLC tensor (n*p^2, m, w^2, c_qk//m)

        # param-free multihead attention
        attn_weight = (
            q_pix * self.scale
        ) @ k_pix_sel  # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight = self.attn_act(attn_weight)
        out = (
            attn_weight @ v_pix_sel
        )  # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
        out = rearrange(
            out,
            "(n j i) m (h w) c -> n (j h) (i w) (m c)",
            j=self.n_win,
            i=self.n_win,
            h=H // self.n_win,
            w=W // self.n_win,
        )

        out = out + lepe
        # output linear
        out = self.wo(out)

        # NOTE: use padding for semantic segmentation
        # crop padded region
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        out = out.permute(0, 3, 1, 2).contiguous()
        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return out
