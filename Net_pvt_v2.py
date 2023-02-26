import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from lavaz_loss import lovasz_hinge2
import math


class Mlp2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block2(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention2(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp2(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4,
                 linear=False,
                 pretrain=None
                 ):
        super().__init__()
        self.pretrain = pretrain
        self.embed_dims = embed_dims

        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block2(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    # https://github.com/whai362/PVT/blob/v2/segmentation/pvt.py
    def forward(self, x):
        outs = []
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            # if i != self.num_stages - 1:
            #	x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict





class pvt_v2_b4(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b4, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            pretrain='pvt_v2_b4.pth',
            **kwargs)


class MixUpSample(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.mixing = nn.Parameter(torch.tensor(0.5))
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.mixing * F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False) \
            + (1 - self.mixing) * F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        return x


def Conv2dBnReLU(in_channel, out_channel, kernel_size=3, padding=1, stride=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation,
                  bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )


class ASPP2(nn.Module):

    def __init__(self,
                 in_channel,
                 channel,
                 dilation,
                 ):
        super(ASPP2, self).__init__()

        self.conv = nn.ModuleList()
        for d in dilation:
            self.conv.append(
                Conv2dBnReLU(
                    in_channel,
                    channel,
                    kernel_size=1 if d == 1 else 3,
                    dilation=d,
                    padding=0 if d == 1 else d,
                )
            )

        self.out = Conv2dBnReLU(
            len(dilation) * channel,
            channel,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        aspp = []
        for conv in self.conv:
            aspp.append(conv(x))
        aspp = torch.cat(aspp, dim=1)
        out = self.out(aspp)
        return out


class DSConv2d(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1
                 ):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DSASPP(nn.Module):

    def __init__(self,
                 in_channel,
                 channel,
                 dilation,
                 ):
        super(DSASPP, self).__init__()

        self.conv = nn.ModuleList()
        for d in dilation:
            if d == 1:
                self.conv.append(
                    Conv2dBnReLU(
                        in_channel,
                        channel,
                        kernel_size=1 if d == 1 else 3,
                        dilation=d,
                        padding=0 if d == 1 else d,
                    )
                )
            else:
                self.conv.append(
                    DSConv2d(
                        in_channel,
                        channel,
                        kernel_size=3,
                        dilation=d,
                        padding=d,
                    )
                )

        self.out = Conv2dBnReLU(
            len(dilation) * channel,
            channel,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        aspp = []
        for conv in self.conv:
            aspp.append(conv(x))
        aspp = torch.cat(aspp, dim=1)
        out = self.out(aspp)
        return out


class DaformerDecoder(nn.Module):
    def __init__(
            self,
            encoder_dim=[32, 64, 160, 256],
            decoder_dim=256,
            dilation=[1, 6, 12, 18],
            use_bn_mlp=True,
            fuse='conv3x3',
    ):
        super().__init__()
        self.mlp = nn.ModuleList([
            nn.Sequential(
                # Conv2dBnReLU(dim, decoder_dim, 1, padding=0), #follow mmseg to use conv-bn-relu
                *(
                    (nn.Conv2d(dim, decoder_dim, 1, padding=0, bias=False),
                     nn.BatchNorm2d(decoder_dim),
                     nn.ReLU(inplace=True),
                     ) if use_bn_mlp else
                    (nn.Conv2d(dim, decoder_dim, 1, padding=0, bias=True),)
                ),

                MixUpSample(2 ** i) if i != 0 else nn.Identity(),
            ) for i, dim in enumerate(encoder_dim)])

        if fuse == 'conv1x1':
            self.fuse = nn.Sequential(
                nn.Conv2d(len(encoder_dim) * decoder_dim, decoder_dim, 1, padding=0, bias=False),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True),
            )

        if fuse == 'conv3x3':
            self.fuse = nn.Sequential(
                nn.Conv2d(len(encoder_dim) * decoder_dim, decoder_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True),
            )

        if fuse == 'aspp':
            self.fuse = ASPP2(
                decoder_dim * len(encoder_dim),
                decoder_dim,
                dilation,
            )

        if fuse == 'ds-aspp':
            self.fuse = DSASPP(
                decoder_dim * len(encoder_dim),
                decoder_dim,
                dilation,
            )

    def forward(self, feature):

        out = []
        for i, f in enumerate(feature):
            f = self.mlp[i](f)
            out.append(f)
        # print(f.shape)
        x = self.fuse(torch.cat(out, dim=1))
        return x, out


class daformer_conv3x3(DaformerDecoder):
    def __init__(self, **kwargs):
        super(daformer_conv3x3, self).__init__(
            fuse='conv3x3',
            **kwargs
        )


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


def criterion_aux_loss(logit, mask):
    mask = F.interpolate(mask, size=logit.shape[-2:], mode='nearest')
    loss = F.binary_cross_entropy_with_logits(logit, mask)
    return loss


class Net7(nn.Module):

    def load_pretrain(self, ):
        checkpoint = '/root/.cache/torch/hub/checkpoints/pvt_v2_b4.pth'
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        # print(list(checkpoint.keys()))
        # ['epoch', 'arch', 'state_dict', 'optimizer', 'version', 'args', 'amp', 'metric']
        state_dict = checkpoint
        self.encoder.load_state_dict(state_dict, strict=False)
        print(self.encoder.load_state_dict(state_dict, strict=False))

    def __init__(self,
                 encoder=pvt_v2_b4,
                 decoder=daformer_conv3x3,
                 encoder_cfg={},
                 decoder_cfg={},
                 ):
        super(Net7, self).__init__()
        decoder_dim = 320

        # ----

        self.rgb = RGB()
        self.output_type = ['inference', 'loss']
        self.encoder = encoder()
        encoder_dim = self.encoder.embed_dims
        # [64, 128, 320, 512]

        self.decoder = decoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
        )
        self.logit = nn.Sequential(
            nn.Conv2d(decoder_dim, 1, kernel_size=1),
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
        logit = self.logit(last)
        logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
        # print(logit.shape)

        output = {}
        if 'loss' in self.output_type:
            output['bce_loss'] = F.binary_cross_entropy_with_logits(logit, batch['mask'])
            # output['bce_loss'] += lovasz_hinge2(logit.view(-1, 768, 768), batch['mask'].view(-1, 768, 768))
            for i in range(4):
                output['aux%d_loss' % i] = criterion_aux_loss(self.aux[i](encoder[i]), batch['mask'])

        if 'inference' in self.output_type:
            output['probability'] = torch.sigmoid(logit)

        return output


if __name__ == '__main__':

    if 1:
        model = Net7()
        # model.load_pretrain()
        input = {}
        input['image'] = torch.rand(1, 3, 768, 768)
        input['mask'] = torch.ones((1, 1, 768, 768))
        output = model(input)
        print(output['probability'].shape)

        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameters: %.2fM" % (total / 1e6))


    if 0:
        net = coat_lite_medium()
        print(net)
        x = torch.rand(2, 3, 800, 800)
        out = net(x)
        print([f.shape for f in out])

        checkpoint = '/root/share1/data/pretrain_model/coat_lite_medium_384x384_f9129688.pth'
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        print(list(checkpoint.keys()))
        # ['epoch', 'arch', 'state_dict', 'optimizer', 'version', 'args', 'amp', 'metric']
        state_dict = checkpoint['model']
        print(net.load_state_dict(state_dict, strict=False))