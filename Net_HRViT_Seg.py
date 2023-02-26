import logging
import math
from functools import lru_cache
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import Tensor
from torch.types import _size


logger = logging.getLogger(__name__)


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "input_size": (3, 768, 768),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "first_conv": "patch_embed.proj",
        **kwargs,
    }


default_cfgs = {
    "hrvit_224": _cfg(),
}


class MixConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv_3 = nn.Conv2d(
            in_channels // 2,
            out_channels // 2,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=groups // 2,
            bias=bias,
        )
        self.conv_5 = nn.Conv2d(
            in_channels - in_channels // 2,
            out_channels - out_channels // 2,
            kernel_size + 2,
            stride=stride,
            padding=padding + 1,
            dilation=dilation,
            groups=groups - groups // 2,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.conv_3(x1)
        x2 = self.conv_5(x2)
        x = torch.cat([x1, x2], dim=1)
        return x


class DES(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        act_func: nn.Module = nn.GELU,
    ) -> None:
        super().__init__()
        _, self.p = self._decompose(min(in_features, out_features))
        self.k_out = out_features // self.p
        self.k_in = in_features // self.p
        self.proj_right = nn.Linear(self.p, self.p, bias=bias)
        self.act = act_func()
        self.proj_left = nn.Linear(self.k_in, self.k_out, bias=bias)

    def _decompose(self, n: int) -> List[int]:
        assert n % 2 == 0, f"Feature dimension has to be a multiple of 2, but got {n}"
        e = int(math.log2(n))
        e1 = e // 2
        e2 = e - e // 2
        return 2 ** e1, 2 ** e2

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[:-1]
        x = x.view(*B, self.k_in, self.p)
        x = self.proj_right(x).transpose(-1, -2)
        if self.act is not None:
            x = self.act(x)
        x = self.proj_left(x).transpose(-1, -2).flatten(-2, -1)

        return x


class MixCFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_func: nn.Module = nn.GELU,
        with_cp: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.with_cp = with_cp
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.conv = MixConv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features,
            dilation=1,
            bias=True,
        )
        self.act = act_func()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        def _inner_forward(x: Tensor) -> Tensor:
            x = self.fc1(x)
            B, N, C = x.shape
            x = self.conv(x.transpose(1, 2).view(B, C, H, W))
            x = self.act(x)
            x = self.fc2(x.flatten(2).transpose(-1, -2))
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


class HRViTAttention(nn.Module):
    def __init__(
        self,
        in_dim: int = 64,
        dim: int = 64,
        heads: int = 2,
        ws: int = 1,  # window size
        qk_scale: Optional[float] = None,
        proj_drop: float = 0.0,
        with_cp: bool = False,
    ):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."
        self.in_dim = in_dim
        self.dim = dim
        self.heads = heads
        self.dim_head = dim // heads
        self.ws = ws
        self.with_cp = with_cp

        self.to_qkv = nn.Linear(in_dim, 2 * dim)

        self.scale = qk_scale or self.dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)

        self.attn_act = nn.Hardswish(inplace=True)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop),
        )

        self.attn_bn = nn.BatchNorm1d(
            dim, momentum=1 - 0.9 ** 0.5 if self.with_cp else 0.1
        )
        nn.init.constant_(self.attn_bn.bias, 0)
        nn.init.constant_(self.attn_bn.weight, 0)

        self.parallel_conv = nn.Sequential(
            nn.Hardswish(inplace=False),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                groups=dim,
            ),
        )

    @lru_cache(maxsize=4)
    def _generate_attn_mask(self, h: int, hp: int, device):
        x = torch.empty(hp, hp, device=device).fill_(-100.0)
        x[:h, :h] = 0
        return x

    def _cross_shaped_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        H: int,
        W: int,
        HP: int,
        WP: int,
        ws: int,
        horizontal: bool = True,
    ):
        B, N, C = q.shape
        if C < self.dim_head:  # half channels are smaller than the defined dim_head
            dim_head = C
            scale = dim_head ** -0.5
        else:
            scale = self.scale
            dim_head = self.dim_head

        if horizontal:
            q, k, v = map(
                lambda y: y.reshape(B, HP // ws, ws, W, C // dim_head, -1)
                .permute(0, 1, 4, 2, 3, 5)
                .flatten(3, 4),
                (q, k, v),
            )
        else:
            q, k, v = map(
                lambda y: y.reshape(B, H, WP // ws, ws, C // dim_head, -1)
                .permute(0, 2, 4, 3, 1, 5)
                .flatten(3, 4),
                (q, k, v),
            )

        attn = q.matmul(k.transpose(-2, -1)).mul(
            scale
        )  # [B,H_2//ws,W_2//ws,h,(b1*b2+1)*(ws*ws),(b1*b2+1)*(ws*ws)]

        ## need to mask zero padding before softmax
        if horizontal and HP != H:
            attn_pad = attn[:, -1:]  # [B, 1, num_heads, ws*W, ws*W]
            mask = self._generate_attn_mask(
                h=(ws - HP + H) * W, hp=attn.size(-2), device=attn.device
            )  # [ws*W, ws*W]
            attn_pad = attn_pad + mask
            attn = torch.cat([attn[:, :-1], attn_pad], dim=1)

        if not horizontal and WP != W:
            attn_pad = attn[:, -1:]  # [B, 1, num_head, ws*H, ws*H]
            mask = self._generate_attn_mask(
                h=(ws - WP + W) * H, hp=attn.size(-2), device=attn.device
            )  # [ws*H, ws*H]
            attn_pad = attn_pad + mask
            attn = torch.cat([attn[:, :-1], attn_pad], dim=1)

        attn = self.attend(attn)

        attn = attn.matmul(v)  # [B,H_2//ws,W_2//ws,h,(b1*b2+1)*(ws*ws),D//h]

        attn = rearrange(
            attn,
            "B H h (b W) d -> B (H b) W (h d)"
            if horizontal
            else "B W h (b H) d -> B H (W b) (h d)",
            b=ws,
        )  # [B,H_1, W_1,D]
        if horizontal and HP != H:
            attn = attn[:, :H, ...]
        if not horizontal and WP != W:
            attn = attn[:, :, :W, ...]
        attn = attn.flatten(1, 2)
        return attn

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        def _inner_forward(x: Tensor) -> Tensor:
            B = x.shape[0]
            ws = self.ws
            qv = self.to_qkv(x)
            q, v = qv.chunk(2, dim=-1)

            v_conv = (
                self.parallel_conv(v.reshape(B, H, W, -1).permute(0, 3, 1, 2))
                .flatten(2)
                .transpose(-1, -2)
            )

            qh, qv = q.chunk(2, dim=-1)
            vh, vv = v.chunk(2, dim=-1)
            kh, kv = vh, vv  # share key and value

            # padding to a multple of window size
            if H % ws != 0:
                HP = int((H + ws - 1) / ws) * ws
                qh = (
                    F.pad(
                        qh.transpose(-1, -2).reshape(B, -1, H, W),
                        pad=[0, 0, 0, HP - H],
                    )
                    .flatten(2, 3)
                    .transpose(-1, -2)
                )
                vh = (
                    F.pad(
                        vh.transpose(-1, -2).reshape(B, -1, H, W),
                        pad=[0, 0, 0, HP - H],
                    )
                    .flatten(2, 3)
                    .transpose(-1, -2)
                )
                kh = vh
            else:
                HP = H

            if W % ws != 0:
                WP = int((W + ws - 1) / ws) * ws
                qv = (
                    F.pad(
                        qv.transpose(-1, -2).reshape(B, -1, H, W),
                        pad=[0, WP - W, 0, 0],
                    )
                    .flatten(2, 3)
                    .transpose(-1, -2)
                )
                vv = (
                    F.pad(
                        vv.transpose(-1, -2).reshape(B, -1, H, W),
                        pad=[0, WP - W, 0, 0],
                    )
                    .flatten(2, 3)
                    .transpose(-1, -2)
                )
                kv = vv
            else:
                WP = W

            attn_h = self._cross_shaped_attention(
                qh,
                kh,
                vh,
                H,
                W,
                HP,
                W,
                ws,
                horizontal=True,
            )
            attn_v = self._cross_shaped_attention(
                qv,
                kv,
                vv,
                H,
                W,
                H,
                WP,
                ws,
                horizontal=False,
            )

            attn = torch.cat([attn_h, attn_v], dim=-1)
            attn = attn.add(v_conv)
            attn = self.attn_act(attn)

            attn = self.to_out(attn)
            attn = self.attn_bn(attn.flatten(0, 1)).view_as(attn)
            return attn

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

    def extra_repr(self) -> str:
        s = f"window_size={self.ws}"
        return s


class HRViTBlock(nn.Module):
    def __init__(
        self,
        in_dim: int = 64,
        dim: int = 64,
        heads: int = 2,
        proj_dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        ws: int = 1,
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        self.with_cp = with_cp

        # build layer normalization
        self.attn_norm = nn.LayerNorm(in_dim)

        # build attention layer
        self.attn = HRViTAttention(
            in_dim=in_dim,
            dim=dim,
            heads=heads,
            ws=ws,
            proj_drop=proj_dropout,
            with_cp=with_cp,
        )

        # build diversity-enhanced shortcut DES
        self.des = DES(
            in_features=in_dim,
            out_features=dim,
            bias=True,
            act_func=nn.GELU,
        )
        # build drop path
        self.attn_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # build layer normalization
        self.ffn_norm = nn.LayerNorm(in_dim)

        # build FFN
        self.ffn = MixCFN(
            in_features=in_dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            act_func=nn.GELU,
            with_cp=with_cp,
        )

        # build drop path
        self.ffn_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        # attention block
        res = x
        x = self.attn_norm(x)
        x = self.attn(x, H, W)
        x_des = self.des(res)
        x = self.attn_drop_path(x.add(x_des)).add(res)

        # ffn block
        res = x
        x = self.ffn_norm(x)
        x = self.ffn(x, H, W)
        x = self.ffn_drop_path(x).add(res)

        return x


class HRViTPatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: _size = 3,
        stride: int = 1,
        dim: int = 64,
    ) -> None:
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.dim = dim

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                dim,
                dim,
                kernel_size=self.patch_size,
                stride=stride,
                padding=(self.patch_size[0] // 2, self.patch_size[1] // 2),
                groups=dim,
            ),
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, int, int]:
        x = self.proj(x)
        H, W = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class HRViTFusionBlock(nn.Module):
    def __init__(
        self,
        in_channels: Tuple[int] = (32, 64, 128, 256),
        out_channels: Tuple[int] = (32, 64, 128, 256),
        act_func: nn.Module = nn.GELU,
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_func = act_func
        self.with_cp = with_cp
        self.n_outputs = len(out_channels)
        self._build_fuse_layers()

    def _build_fuse_layers(self):
        self.blocks = nn.ModuleList([])
        n_inputs = len(self.in_channels)
        for i, outc in enumerate(self.out_channels):
            blocks = nn.ModuleList([])

            start = 0
            end = n_inputs
            for j in range(start, end):
                inc = self.in_channels[j]
                if j == i:
                    blocks.append(nn.Identity())
                elif j < i:
                    block = [
                        nn.Conv2d(
                            inc,
                            inc,
                            kernel_size=2 ** (i - j) + 1,
                            stride=2 ** (i - j),
                            dilation=1,
                            padding=2 ** (i - j) // 2,
                            groups=inc,
                            bias=False,
                        ),
                        nn.BatchNorm2d(inc),
                        nn.Conv2d(
                            inc,
                            outc,
                            kernel_size=1,
                            stride=1,
                            dilation=1,
                            padding=0,
                            groups=1,
                            bias=True,
                        ),
                        nn.BatchNorm2d(outc),
                    ]

                    blocks.append(nn.Sequential(*block))

                else:
                    block = [
                        nn.Conv2d(
                            inc,
                            outc,
                            kernel_size=1,
                            stride=1,
                            dilation=1,
                            padding=0,
                            groups=1,
                            bias=True,
                        ),
                        nn.BatchNorm2d(outc),
                    ]

                    block.append(
                        nn.Upsample(
                            scale_factor=2 ** (j - i),
                            mode="nearest",
                        ),
                    )
                    blocks.append(nn.Sequential(*block))
            self.blocks.append(blocks)

        self.act = nn.ModuleList([self.act_func() for _ in self.out_channels])

    def forward(
        self,
        x: Tuple[
            Tensor,
        ],
    ) -> Tuple[Tensor,]:

        out = [None] * len(self.blocks)
        n_inputs = len(x)

        for i, (blocks, act) in enumerate(zip(self.blocks, self.act)):
            start = 0
            end = n_inputs
            for j, block in zip(range(start, end), blocks):
                out[i] = block(x[j]) if out[i] is None else out[i] + block(x[j])
            out[i] = act(out[i])

        return out


class HRViTStem(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size = 3,
        stride: _size = 4,
        dilation: _size = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        stride = (stride[0]//2, stride[1]//2)
        dilation = to_2tuple(dilation)
        # same padding
        padding = [
            (dilation[i] * (kernel_size[i] - 1) + 1) // 2
            for i in range(len(kernel_size))
        ]


        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


class HRViTStage(nn.Module):
    def __init__(
        self,
        #### Patch Embed Config ####
        in_channels: Tuple[
            int,
        ] = (32, 64, 128, 256),
        out_channels: Tuple[
            int,
        ] = (32, 64, 128, 256),
        block_list: Tuple[
            int,
        ] = (1, 1, 6, 2),
        #### HRViTAttention Config ####
        dim_head: int = 32,
        ws_list: Tuple[
            int,
        ] = (1, 2, 7, 7),
        proj_dropout: float = 0.0,
        drop_path_rates: Tuple[float] = (
            0.0,
        ),  # different droprate for different attn/mlp
        #### MixCFN Config ####
        mlp_ratio_list: Tuple[
            int,
        ] = (4, 4, 4, 4),
        dropout: float = 0.0,
        #### Gradient Checkpointing #####
        with_cp: bool = False,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.ModuleList(
            [
                HRViTPatchEmbed(
                    in_channels=inc,
                    patch_size=3,
                    stride=1,
                    dim=outc,
                )
                for inc, outc in zip(in_channels, out_channels)
            ]
        )  # one patch embedding for each branch

        ## we arrange blocks in stages/layers
        n_inputs = len(out_channels)

        self.branches = nn.ModuleList([])
        for i, n_blocks in enumerate(block_list[:n_inputs]):
            blocks = []
            for j in range(n_blocks):
                blocks += [
                    HRViTBlock(
                        in_dim=out_channels[i],
                        dim=out_channels[i],
                        heads=out_channels[i] // dim_head,  # automatically derive heads
                        proj_dropout=proj_dropout,
                        mlp_ratio=mlp_ratio_list[i],
                        drop_path=drop_path_rates[j],
                        ws=ws_list[i],
                        with_cp=with_cp,
                    )
                ]

            blocks = nn.ModuleList(blocks)
            self.branches.append(blocks)
        self.norm = nn.ModuleList([nn.LayerNorm(outc) for outc in out_channels])

    def forward(
        self,
        x: Tuple[
            Tensor,
        ],
    ) -> Tuple[Tensor,]:
        B = x[0].shape[0]
        x = list(x)
        H, W = [], []
        ## patch embed
        for i, (xx, embed) in enumerate(zip(x, self.patch_embed)):
            xx, h, w = embed(xx)
            x[i] = xx
            H.append(h)
            W.append(w)

        ## HRViT blocks
        for i, (branch, h, w) in enumerate(zip(self.branches, H, W)):
            for block in branch:
                x[i] = block(x[i], h, w)

        ## LN at the end of each stage
        for i, (xx, norm, h, w) in enumerate(zip(x, self.norm, H, W)):
            xx = norm(xx)
            xx = xx.reshape(B, h, w, -1).permute(0, 3, 1, 2).contiguous()
            x[i] = xx
        return x


class HRViT(nn.Module):
    def __init__(
        self,
        #### HRViT Stem Config ####
        in_channels: int = 3,
        stride: int = 4,
        channels: int = 64,
        #### Branch Config ####
        channel_list: Tuple[Tuple[int,],] = (
            (32,),
            (32, 64),
            (32, 64, 128),
            (32, 64, 128),
            (32, 64, 128, 256),
            (32, 64, 128, 256),
        ),
        block_list: Tuple[Tuple[int]] = (
            (1,),
            (1, 1),
            (1, 1, 6),
            (1, 1, 6),
            (1, 1, 6, 2),
            (1, 1, 2, 2),
        ),
        #### HRViTAttention Config ####
        dim_head: int = 32,
        ws_list: Tuple[
            int,
        ] = (1, 2, 7, 7),
        proj_dropout: float = 0.0,
        drop_path_rate: float = 0.0,  # different droprate for different attn/mlp
        #### HRViTFeedForward Config ####
        mlp_ratio_list: Tuple[
            int,
        ] = (4, 4, 4, 4),
        dropout: float = 0.0,
        #### Gradient Checkpointing #####
        with_cp: bool = False,
    ) -> None:
        super().__init__()

        self.features = []
        self.ws_list = ws_list
        self.with_cp = with_cp

        # calculate drop path rates
        total_blocks = sum(max(b) for b in block_list)

        total_drop_path_rates = (
            torch.linspace(0, drop_path_rate, total_blocks).numpy().tolist()
        )

        cur = 0
        self.channel_list = channel_list = [[channels]] + list(channel_list)

        # build stem
        self.stem = HRViTStem(
            in_channels=in_channels, out_channels=channels, kernel_size=3, stride=4
        )

        # build backbone
        for i, blocks in enumerate(block_list):
            inc, outc = channel_list[i : i + 2]
            depth_per_stage = max(blocks)

            self.features.extend(
                [
                    HRViTFusionBlock(
                        in_channels=inc,
                        out_channels=inc
                        if len(inc) == len(outc)
                        else list(inc) + [outc[-1]],
                        act_func=nn.GELU,
                        with_cp=False,
                    ),
                    HRViTStage(
                        #### Patch Embed Config ####
                        in_channels=inc
                        if len(inc) == len(outc)
                        else list(inc) + [outc[-1]],
                        out_channels=outc,
                        block_list=blocks,
                        dim_head=dim_head,
                        #### HRViTBlock Config ####
                        ws_list=ws_list,
                        proj_dropout=proj_dropout,
                        drop_path_rates=total_drop_path_rates[
                            cur : cur + depth_per_stage
                        ],  # different droprate for different attn/mlp
                        #### MixCFN Config ####
                        mlp_ratio_list=mlp_ratio_list,
                        dropout=dropout,
                        #### Gradient Checkpointing #####
                        with_cp=with_cp,
                    ),
                ]
            )
            cur += depth_per_stage

        self.features = nn.Sequential(*self.features)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)


    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        if self.num_classes != num_classes:
            logger.info("Reset head to", num_classes)
            self.num_classes = num_classes

    def forward_features(
        self, x: Tensor
    ) -> Tuple[Tensor,]:
        # stem
        x = self.stem(x)
        # backbone
        x = self.features((x,))
        return x

    def forward(self, x: Tensor) -> Tensor:
        # stem and backbone
        x = self.forward_features(x)
        # # classifier
        # x = self.head(x)
        return x


def HRViT_b1_224(pretrained=False, **kwargs):
    model = HRViT(
        stride=4,
        channels=64,
        channel_list=(
            (32,),
            (32, 64),
            (32, 64, 128),
            (32, 64, 128),
            (32, 64, 128, 256),
            (32, 64, 128, 256),
        ),
        block_list=(
            (1,),
            (1, 1),
            (1, 1, 6),
            (1, 1, 6),
            (1, 1, 6, 2),
            (1, 1, 2, 2),
        ),
        dim_head=32,
        ws_list=(1, 2, 7, 7),
        proj_dropout=0.0,
        # drop_path_rate=0.1,
        mlp_ratio_list=(4, 4, 4, 4),
        dropout=0.0,
        # head_dropout=0.1,
        **kwargs,
    )
    model.default_cfg = default_cfgs["hrvit_224"]
    return model


def HRViT_b2_224(pretrained=False, **kwargs):
    model = HRViT(
        stride=4,
        channels=64,
        channel_list=(
            (48,),
            (48, 96),
            (48, 96, 240),
            (48, 96, 240),
            (48, 96, 240, 384),
            (48, 96, 240, 384),
        ),
        block_list=(
            (1,),
            (1, 1),
            (1, 1, 6),
            (1, 1, 6),
            (1, 1, 6, 2),
            (1, 1, 6, 2),
        ),
        dim_head=24,
        ws_list=(1, 2, 7, 7),
        proj_dropout=0.0,
        drop_path_rate=0.1,
        mlp_ratio_list=(2, 3, 3, 3),
        dropout=0.0,
        # head_dropout=0.1,
        **kwargs,
    )
    model.default_cfg = default_cfgs["hrvit_224"]
    return model


def HRViT_b3_224(pretrained=False, **kwargs):
    model = HRViT(
        stride=4,
        channels=64,
        channel_list=(
            (64,),
            (64, 128),
            (64, 128, 256),
            (64, 128, 256),
            (64, 128, 256, 512),
            (64, 128, 256, 512),
        ),
        block_list=(
            (1,),
            (1, 1),
            (1, 1, 6),
            (1, 1, 6),
            (1, 1, 6, 3),
            (1, 1, 6, 3),
        ),
        dim_head=32,
        ws_list=(1, 2, 7, 7),
        proj_dropout=0.0,
        # drop_path_rate=0.1,
        mlp_ratio_list=(2, 2, 2, 2),
        dropout=0.0,
        # head_dropout=0.1,
        **kwargs,
    )
    model.default_cfg = default_cfgs["hrvit_224"]
    return model


class MixUpSample(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.mixing = nn.Parameter(torch.tensor(0.5))
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.mixing * F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False) \
            + (1 - self.mixing) * F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        return x


class SegformerDecoder(nn.Module):
    def __init__(
            self,
            encoder_dim=[64, 128, 320, 512],
            decoder_dim=256,
    ):
        super().__init__()
        self.mixing = nn.Parameter(torch.FloatTensor([0.5, 0.5, 0.5, 0.5]))
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, decoder_dim, 1, padding=0, bias=False),  # follow mmseg to use conv-bn-relu
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True),
                MixUpSample(2 ** i) if i != 0 else nn.Identity(),
            ) for i, dim in enumerate(encoder_dim)])

        self.fuse = nn.Sequential(
            nn.Conv2d(len(encoder_dim) * decoder_dim, decoder_dim, 1, padding=0, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            # nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, bias=False),
            # nn.BatchNorm2d(decoder_dim),
            # nn.ReLU(inplace=True),
        )

    def forward(self, feature):
        out = []
        for i, f in enumerate(feature):
            f = self.mlp[i](f)
            out.append(f)

        x = self.fuse(torch.cat(out, dim=1))
        return x, out


class RGB(nn.Module):
    IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]  # [0.5, 0.5, 0.5]
    IMAGE_RGB_STD = [0.229, 0.224, 0.225]  # [0.5, 0.5, 0.5]

    def __init__(self, ):
        super(RGB, self).__init__()
        self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
        self.register_buffer('std', torch.ones(1, 3, 1, 1))
        self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
        self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return x


class Net6(nn.Module):
    # def load_pretrain(self, ):
    #     checkpoint = cfg[self.arch]['checkpoint']
    #     print('load %s' % checkpoint)
    #     checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)  # True
    #     print(self.encoder.load_state_dict(checkpoint, strict=False))  # True

    def __init__(self, ):
        super(Net6, self).__init__()
        self.output_type = ['inference', 'loss']
        self.rgb = RGB()
        self.dropout = nn.Dropout(0.1)

        self.arch = 'mit_b2'
        self.encoder = HRViT_b2_224()
        encoder_dim = [48, 96, 240, 384]
        # [64, 128, 320, 512]

        self.decoder = SegformerDecoder(
            encoder_dim=encoder_dim,
            decoder_dim=240,
        )
        self.logit = nn.Sequential(
            nn.Conv2d(240, 1, kernel_size=1, padding=0),
        )
        self.aux = nn.ModuleList([
            nn.Conv2d(encoder_dim[i], 1, kernel_size=1, padding=0) for i in range(4)
        ])

    def forward(self, batch):

        x = batch['image']
        x = self.rgb(x)

        B, C, H, W = x.shape
        encoder = self.encoder(x)
        # print([f.shape for f in encoder])

        last, decoder = self.decoder(encoder)
        last = self.dropout(last)
        logit = self.logit(last)
        logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
        # print(logit.shape)

        output = {}
        if 'loss' in self.output_type:
            output['bce_loss'] = F.binary_cross_entropy_with_logits(logit, batch['mask'])
            for i in range(4):
                output['aux%d_loss' % i] = criterion_aux_loss(self.aux[i](encoder[i]), batch['mask'])

        if 'inference' in self.output_type:
            output['probability'] = torch.sigmoid(logit)

        return output


def criterion_aux_loss(logit, mask):
    mask = F.interpolate(mask, size=logit.shape[-2:], mode='nearest')
    loss = F.binary_cross_entropy_with_logits(logit, mask)
    return loss


if __name__ == '__main__':
    if 0:
        model = HRViT_b2_224()
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameters: %.2fM" % (total / 1e6))
    if 1:
        model = Net6()
        # model.load_pretrain()
        input = {}
        input['image'] = torch.rand(2, 3, 768, 768)
        input['mask'] = torch.ones((2, 1, 768, 768))
        output = model(input)

        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameters: %.2fM" % (total / 1e6))