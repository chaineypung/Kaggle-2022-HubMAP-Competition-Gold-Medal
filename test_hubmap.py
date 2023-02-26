import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
import cv2
import os
import gc
from tqdm import tqdm
import rasterio
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from rasterio.windows import Window
from torch.utils.data import Dataset, DataLoader
import warnings
from functools import partial
import cv2
import os
import gc
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import warnings
import torch
import os.path
import torch.cuda.amp as amp
import tifffile
import math
from skimage import measure
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
warnings.filterwarnings("ignore")


is_amp = True
bs = 4
sz = 256
sz_reduction = 2
expansion = 1024
TH = 0.5
s_th = 40
p_th = 1000 * (sz // 256) ** 2
identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
mean = np.array([0.7720342, 0.74582646, 0.76392896])
std = np.array([0.24745085, 0.26182273, 0.25782376])

IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD = [0.229, 0.224, 0.225]

DATA = r'/root/autodl-tmp/spleen/img'
mask_dir = r'/root/autodl-tmp/spleen/mask'

df_sample = pd.read_csv(r'/root/autodl-tmp/spleen/train.csv')

img_txt = r'/root/HuBMAP/code/train_valid_list_path_spleen/train_image.txt'
mask_txt = r'/root/HuBMAP/code/train_valid_list_path_spleen/train_mask.txt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS3 = [r'/root/HuBMAP/code/saved_model/weights/best_model_segbit2v2_1_7949.pth', \
           r'/root/HuBMAP/code/saved_model/weights/best_model_segbit2v2_2_7965.pth', \
           r'/root/HuBMAP/code/saved_model/weights/best_model_segbit2v2_5_7901.pth']
MODELS4 = [r'/root/HuBMAP/code/saved_model/weights/best_model_upernet_1.pth', \
           r'/root/HuBMAP/code/saved_model/weights/best_model_upernet_2.pth']

MODELS_nolung_3 = [r'/root/HuBMAP/code/saved_model/weights/best_model_segbit2v2_1_7949.pth', \
           r'/root/HuBMAP/code/saved_model/weights/best_model_segbit2v2_2_7965.pth', \
           r'/root/HuBMAP/code/saved_model/weights/best_model_segbit2v2_5_7901.pth']
MODELS_nolung_4 = [r'/root/HuBMAP/code/saved_model/weights/best_model_upernet_1.pth', \
           r'/root/HuBMAP/code/saved_model/weights/best_model_upernet_2.pth']


def read_tiff(path, scale=None,
              verbose=0):  # Modified from https://www.kaggle.com/code/abhinand05/hubmap-extensive-eda-what-are-we-hacking
    image = tifffile.imread(path)
    if len(image.shape) == 5:
        image = image.squeeze().transpose(1, 2, 0)
    if verbose:
        print(f"[{path}] Image shape: {image.shape}")
    if scale:
        new_size = (image.shape[1] // scale, image.shape[0] // scale)
        image = cv2.resize(image, new_size)
        if verbose:
            print(f"[{path}] Resized Image shape: {image.shape}")
    return image


def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if isinstance(enc, np.float) and np.isnan(enc): continue
        s = enc.split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            img[start:start + length] = 1 + m
    return img.reshape(shape).T


def mask2enc(mask, n=1):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1, n + 1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0:
            encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs


def rle_encode_less_memory(img):
    # the image should be transposed
    pixels = img.T.flatten()

    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)


def NN_interpolation(img, dstH, dstW):
    scrH, scrW = img.shape
    retimg = np.zeros((dstH, dstW), dtype=np.uint8)
    for i in range(dstH - 1):
        for j in range(dstW - 1):
            scrx = round(i * (scrH / dstH))
            scry = round(j * (scrW / dstW))
            retimg[i, j] = img[scrx, scry]
    return retimg


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2: img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


def image_to_tensor(image, mode='bgr'):  # image mode
    if mode == 'bgr':
        image = image[:, :, ::-1]
    x = image
    x = x.transpose(2, 0, 1)
    x = np.ascontiguousarray(x)
    x = torch.tensor(x, dtype=torch.float)
    return x


class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, sz_reduction=sz_reduction, expansion=expansion):
        self.data = rasterio.open(os.path.join(DATA, idx + '.tiff'), transform=identity,
                                  num_threads='all_cpus')
        # some images have issues with their format
        # and must be saved correctly before reading with rasterio
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        self.shape = self.data.shape
        self.sz_reduction = sz_reduction
        self.sz = sz_reduction * sz
        self.expansion = sz_reduction * expansion
        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        if self.shape[0] < 250:
            if self.data.count == 3:
                img = np.moveaxis(self.data.read([1, 2, 3]), 0, -1)
            else:
                for i, layer in enumerate(self.layers):
                    img = layer.read(1)
            img = cv2.resize(img, (768, 768), interpolation=cv2.INTER_AREA)
            return img2tensor((img / 255.0 - mean) / std), idx
        # the code below may be a little bit difficult to understand,
        # but the thing it does is mapping the original image to
        # tiles created with adding padding, as done in
        # https://www.kaggle.com/iafoss/256x256-images ,
        # and then the tiles are loaded with rasterio
        # n0,n1 - are the x and y index of the tile (idx = n0*self.n1max + n1)
        else:
            n0, n1 = idx // self.n1max, idx % self.n1max
            # x0,y0 - are the coordinates of the lower left corner of the tile in the image
            # negative numbers correspond to padding (which must not be loaded)
            x0, y0 = -self.pad0 // 2 + n0 * self.sz - self.expansion // 2, -self.pad1 // 2 + n1 * self.sz - self.expansion // 2
            # make sure that the region to read is within the image
            p00, p01 = max(0, x0), min(x0 + self.sz + self.expansion, self.shape[0])
            p10, p11 = max(0, y0), min(y0 + self.sz + self.expansion, self.shape[1])
            img = np.zeros((self.sz + self.expansion, self.sz + self.expansion, 3), np.uint8)
            # mapping the loade region to the tile
            if self.data.count == 3:
                img[(p00 - x0):(p01 - x0), (p10 - y0):(p11 - y0)] = np.moveaxis(
                    self.data.read([1, 2, 3], window=Window.from_slices((p00, p01), (p10, p11))), 0, -1)
            else:
                for i, layer in enumerate(self.layers):
                    img[(p00 - x0):(p01 - x0), (p10 - y0):(p11 - y0), i] = \
                        layer.read(1, window=Window.from_slices((p00, p01), (p10, p11)))
            if self.sz_reduction != 1:
                img = cv2.resize(img, (
                    (self.sz + self.expansion) // self.sz_reduction, (self.sz + self.expansion) // self.sz_reduction),
                                 interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            if (s > s_th).sum() <= p_th or img.sum() <= p_th:
                return img2tensor((img / 255.0 - mean) / std), -1
            else:
                return img2tensor((img / 255.0 - mean) / std), idx


class HuBMAPDatasetLarge(Dataset):
    def __init__(self, idx, shape=None, sz=sz, sz_reduction=sz_reduction, expansion=expansion):
        self.data = rasterio.open(os.path.join(DATA, idx + '.tiff'), transform=identity,
                                  num_threads='all_cpus')
        # some images have issues with their format
        # and must be saved correctly before reading with rasterio
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        self.shapetrue = self.data.shape
        self.shape = shape
        self.sz_reduction = sz_reduction
        self.sz = sz_reduction * sz
        self.expansion = sz_reduction * expansion
        self.pad0 = (self.sz - self.shape % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape % self.sz) % self.sz
        self.n0max = (self.shape + self.pad0) // self.sz
        self.n1max = (self.shape + self.pad1) // self.sz

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):

        if self.data.count == 3:
            pic = np.moveaxis(self.data.read([1, 2, 3]), 0, -1)
        else:
            for i, layer in enumerate(self.layers):
                pic = layer.read(1)
        pic = cv2.resize(pic, (self.shape, self.shape), interpolation=cv2.INTER_AREA)

        n0, n1 = idx // self.n1max, idx % self.n1max

        x0, y0 = -self.pad0 // 2 + n0 * self.sz - self.expansion // 2, -self.pad1 // 2 + n1 * self.sz - self.expansion // 2
        # make sure that the region to read is within the image
        p00, p01 = max(0, x0), min(x0 + self.sz + self.expansion, self.shape)
        p10, p11 = max(0, y0), min(y0 + self.sz + self.expansion, self.shape)
        img = np.zeros((self.sz + self.expansion, self.sz + self.expansion, 3), np.uint8)
        # mapping the loade region to the tile

        img[(p00 - x0):(p01 - x0), (p10 - y0):(p11 - y0)] = pic[p00:p01, p10:p11]

        if self.sz_reduction != 1:
            img = cv2.resize(img, (
                (self.sz + self.expansion) // self.sz_reduction, (self.sz + self.expansion) // self.sz_reduction),
                             interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if (s > s_th).sum() <= p_th or img.sum() <= p_th:
            return img2tensor((img / 255.0 - mean) / std), -1
        else:
            return img2tensor((img / 255.0 - mean) / std), idx


class Model_pred:
    def __init__(self, models, dl, dsnmax, tta: bool = True, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half
        self.dsnmax = dsnmax

    def __iter__(self):
        count = 0
        with torch.no_grad():
            for x, y in iter(self.dl):
                if ((y >= 0).sum() > 0):  # exclude empty images
                    x = x[y >= 0].to(device)
                    y = y[y >= 0]
                    if self.half: x = x.half()
                    py = None
                    for model in self.models:
                        p = model(x)
                        # x = x.cpu().numpy()
                        # plt.imshow(x[0, 0, :, :])
                        # plt.show()
                        p = torch.sigmoid(p).detach()
                        if py is None:
                            py = p
                        else:
                            py += p
                    if self.tta:
                        # x,y,xy flips as TTA
                        flips = [[-1], [-2], [-2, -1]]
                        for f in flips:
                            xf = torch.flip(x, f)
                            for model in self.models:
                                p = model(xf)
                                p = torch.flip(p, f)
                                py += torch.sigmoid(p).detach()
                        py /= (1 + len(flips))

                    if self.dsnmax < 250:
                        py = py.permute(0, 2, 3, 1).float().cpu()
                        batch_size = len(py)
                        for i in range(batch_size):
                            yield py[i], y[i]
                            count += 1
                    else:
                        py = F.upsample(py, scale_factor=sz_reduction, mode="bilinear")
                        py = py.permute(0, 2, 3, 1).float().cpu()
                        batch_size = len(py)
                        for i in range(batch_size):
                            yield py[i, expansion * sz_reduction // 2:-expansion * sz_reduction // 2,
                                  expansion * sz_reduction // 2:-expansion * sz_reduction // 2], y[i]
                            count += 1

    def __len__(self):
        return len(self.dl.dataset)


class HuBMAPDatasetseg(Dataset):
    def __init__(self, idx, resizeshape=None, rethick=None):
        self.data = cv2.imread(os.path.join(DATA, idx), cv2.IMREAD_COLOR)
        self.shape = self.data.shape
        self.resizeshape = resizeshape
        self.rethick = rethick

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.rethick == 'kidney':
            image_hsv = cv2.cvtColor(self.data, cv2.COLOR_BGR2HSV).astype(np.float32)
            image_hsv[:, :, 1] *= (1 + (0.15 * -6))
            image_hsv[:, :, 2] *= (1 - (0.15 * -6))
            image_hsv = image_hsv.astype(np.uint8)
            self.data = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
        self.data = self.data.astype(np.float32) / 255
        img = cv2.resize(self.data, (self.resizeshape, self.resizeshape), interpolation=cv2.INTER_AREA)
        return image_to_tensor(img), idx


class HuBMAPDatasetprostate(Dataset):
    def __init__(self, idx, resizeshape=None, rethick=None):
        self.data = cv2.cvtColor(read_tiff(os.path.join(DATA, idx + '.tiff')), cv2.COLOR_RGB2BGR)
        self.shape = self.data.shape
        self.resizeshape = resizeshape
        self.rethick = rethick

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.rethick == 'kidney':
            image_hsv = cv2.cvtColor(self.data, cv2.COLOR_BGR2HSV).astype(np.float32)
            image_hsv[:, :, 1] *= (1 + (0.15 * -6))
            image_hsv[:, :, 2] *= (1 - (0.15 * -6))
            image_hsv = image_hsv.astype(np.uint8)
            self.data = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
        self.data = (self.data.astype(np.float32) / 255 - IMAGE_RGB_MEAN) / IMAGE_RGB_STD
        img = cv2.resize(self.data, (self.resizeshape, self.resizeshape), interpolation=cv2.INTER_AREA)
        return image_to_tensor(img), idx


class Model_predseg:
    def __init__(self, models, dl, dsnmax, tta: bool = True, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half
        self.dsnmax = dsnmax

    def __iter__(self):
        count = 0
        with torch.no_grad():
            with amp.autocast(enabled=is_amp):
                for x, y in iter(self.dl):
                    if ((y >= 0).sum() > 0):  # exclude empty images
                        x = x[y >= 0].to(device)
                        y = y[y >= 0]
                        if self.half: x = x.half()
                        py = None
                        for model in self.models:
                            model.output_type = ['inference']
                            input = {'image': x, 'mask': torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])).to(device)}
                            output = model(input)
                            p = F.interpolate(output['probability'], size=(self.dsnmax, self.dsnmax), mode='bilinear',
                                              align_corners=False).detach()
                            if py is None:
                                py = p
                            else:
                                py += p
                        if self.tta:
                            # x,y,xy flips as TTA
                            flips = [[-1], [-2], [-2, -1]]
                            for f in flips:
                                xf = torch.flip(x, f)
                                for model in self.models:
                                    model.output_type = ['inference']
                                    input = {'image': xf,
                                             'mask': torch.ones((xf.shape[0], 1, xf.shape[2], xf.shape[3])).to(device)}
                                    p = model(input)
                                    p = F.interpolate(p['probability'], size=(self.dsnmax, self.dsnmax),
                                                      mode='bilinear', align_corners=False).detach()
                                    p = torch.flip(p, f)
                                    py += p.detach()
                            py /= (1 + len(flips))

                        py = py.permute(0, 2, 3, 1).float().cpu()
                        batch_size = len(py)
                        for i in range(batch_size):
                            yield py[i], y[i]
                            count += 1

    def __len__(self):
        return len(self.dl.dataset)


class Model_predsegprostate:
    def __init__(self, models, dl, dsnmax, tta: bool = True, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half
        self.dsnmax = dsnmax

    def __iter__(self):
        count = 0
        with torch.no_grad():
            with amp.autocast(enabled=is_amp):
                for x, y in iter(self.dl):
                    if ((y >= 0).sum() > 0):  # exclude empty images
                        x = x[y >= 0].to(device)
                        y = y[y >= 0]
                        if self.half: x = x.half()
                        py = None
                        for model in self.models:
                            p = model(x)
                            p = torch.sigmoid(p)
                            p = F.interpolate(p, size=(self.dsnmax, self.dsnmax), mode='bilinear',
                                              align_corners=False).detach()
                            if py is None:
                                py = p
                            else:
                                py += p
                        if self.tta:
                            # x,y,xy flips as TTA
                            flips = [[-1], [-2], [-2, -1]]
                            for f in flips:
                                xf = torch.flip(x, f)
                                for model in self.models:
                                    p = model(xf)
                                    p = torch.sigmoid(p)
                                    p = F.interpolate(p, size=(self.dsnmax, self.dsnmax), mode='bilinear',
                                                      align_corners=False).detach()
                                    p = torch.flip(p, f)
                                    py += p
                            py /= (1 + len(flips))

                        py = py.permute(0, 2, 3, 1).float().cpu()
                        batch_size = len(py)
                        for i in range(batch_size):
                            yield py[i], y[i]
                            count += 1

    def __len__(self):
        return len(self.dl.dataset)


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


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

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
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
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

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

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

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
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


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

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

    def init_weights(self, pretrained=None):
        pass

    # if isinstance(pretrained, str):
    # 	logger = get_root_logger()
    # 	load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

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

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups=1):
        super().__init__()
        self.depth = nn.Conv2d(inplanes, inplanes, kernel_size=kernel_size,
                               stride=1, padding=padding, dilation=dilation, bias=False, groups=groups)
        self.point = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[6, 12, 18, 24], out_c=None):
        super().__init__()
        self.aspps = [_ASPPModule(inplanes, mid_c, 1, padding=0, dilation=1)] + \
                     [_ASPPModule(inplanes, mid_c, 3, padding=d, dilation=d, groups=4) for d in dilations]
        self.aspps = nn.ModuleList(self.aspps)
        self.global_pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                                         nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
                                         nn.BatchNorm2d(mid_c), nn.ReLU())
        out_c = out_c if out_c is not None else mid_c
        self.out_conv = nn.Sequential(nn.Conv2d(2304, out_c, 3, 1, 1, bias=False),
                                      nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(mid_c * (2 + len(dilations)), out_c, 1, bias=False)
        self._init_weight()

    def forward(self, x):
        x0 = x
        xs = [aspp(x) for aspp in self.aspps]
        x0 = F.interpolate(x0, size=xs[0].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x0] + xs, dim=1)
        return self.out_conv(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class mit_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b1(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b3(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b4(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


class mit_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


pretrain_dir = '/root/.cache/torch/hub/checkpoints'
if 1:
    cfg1 = dict(
        mit_b0=dict(
            checkpoint=pretrain_dir + '/mit_b0.pth',
            builder=mit_b0,
        ),
        mit_b1=dict(
            checkpoint=pretrain_dir + '/mit_b1.pth',
            builder=mit_b1,
        ),
        mit_b2=dict(
            checkpoint=pretrain_dir + '/segformer_b2_backbone_imagenet-3c162bb8.pth',
            builder=mit_b2,
        ),
        mit_b3=dict(
            checkpoint=pretrain_dir + '/mit_b3.pth',
            builder=mit_b3,
        ),

    )


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


class DAformerDecoder(nn.Module):
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
                MixUpSample(2 ** i) if i != 0 else nn.Identity(),
            ) for i, dim in enumerate(encoder_dim)])

        self.aspp = ASPP(1024, 256, out_c=256, dilations=[1, 6, 12, 18])

    def forward(self, feature):
        out = []
        for i, f in enumerate(feature):
            f = self.mlp[i](f)
            out.append(f)

        x = self.aspp(torch.cat(out, dim=1))
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


class Net(nn.Module):
    def load_pretrain(self, ):
        checkpoint = cfg1[self.arch]['checkpoint']
        print('load %s' % checkpoint)
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)  # True
        print(self.encoder.load_state_dict(checkpoint, strict=False))  # True

    def __init__(self, ):
        super(Net, self).__init__()
        self.output_type = ['inference', 'loss']
        self.rgb = RGB()
        self.dropout = nn.Dropout(0.1)

        self.arch = 'mit_b2'
        self.encoder = cfg1[self.arch]['builder']()
        encoder_dim = self.encoder.embed_dims
        # [64, 128, 320, 512]

        self.decoder = SegformerDecoder(
            encoder_dim=encoder_dim,
            decoder_dim=320,
        )
        self.logit = nn.Sequential(
            nn.Conv2d(320, 1, kernel_size=1, padding=0),
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


class Mlp1(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** (-0.5)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], self.num_heads)
        # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp1(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W, mask_matrix):

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift ---
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift ---
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B, H/2, W/2, 4*C
        x = x.view(B, -1, 4 * C)  # B, H/2*W/2, 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 # use_checkpoint=False,
                 ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA ----
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        # ------

        for blk in self.blocks:
            x = blk(x, H, W, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None
                 ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape

        # padding
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x


class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SwinTransformerV1(nn.Module):
    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 out_norm=nn.Identity,  # use nn.Identity, nn.BatchNorm2d, LayerNorm2d
                 **kwargs
                 ):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = np.linspace(0, drop_path_rate, sum(depths)).tolist()  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i < self.num_layers - 1) else None,
            )
            self.layers.append(layer)

        # ---
        # add a norm layer for each output
        self.out_norm = nn.ModuleList(
            [out_norm(int(embed_dim * 2 ** i)) for i in range(self.num_layers)]
        )

        # ---
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)

        # positional encode?
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            x_out, H, W, x, Wh, Ww = self.layers[i](x, Wh, Ww)
            out = x_out.view(-1, H, W, int(self.embed_dim * 2 ** i)).permute(0, 3, 1, 2).contiguous()
            out = self.out_norm[i](out)
            outs.append(out)

        return outs


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class UPerDecoder(nn.Module):
    def __init__(self,
                 in_dim=[256, 512, 1024, 2048],
                 ppm_pool_scale=[1, 2, 3, 6],
                 ppm_dim=512,
                 fpn_out_dim=256
                 ):
        super(UPerDecoder, self).__init__()

        # PPM ----
        dim = in_dim[-1]
        ppm_pooling = []
        ppm_conv = []

        for scale in ppm_pool_scale:
            ppm_pooling.append(
                nn.AdaptiveAvgPool2d(scale)
            )
            ppm_conv.append(
                nn.Sequential(
                    nn.Conv2d(dim, ppm_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(ppm_dim),
                    nn.ReLU(inplace=True)
                )
            )
        self.ppm_pooling = nn.ModuleList(ppm_pooling)
        self.ppm_conv = nn.ModuleList(ppm_conv)
        self.ppm_out = conv3x3_bn_relu(dim + len(ppm_pool_scale) * ppm_dim, fpn_out_dim, 1)

        # FPN ----
        fpn_in = []
        for i in range(0, len(in_dim) - 1):  # skip the top layer
            fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(in_dim[i], fpn_out_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(fpn_out_dim),
                    nn.ReLU(inplace=True)
                )
            )
        self.fpn_in = nn.ModuleList(fpn_in)

        fpn_out = []
        for i in range(len(in_dim) - 1):  # skip the top layer
            fpn_out.append(
                conv3x3_bn_relu(fpn_out_dim, fpn_out_dim, 1),
            )
        self.fpn_out = nn.ModuleList(fpn_out)

        self.fpn_fuse = nn.Sequential(
            conv3x3_bn_relu(len(in_dim) * fpn_out_dim, fpn_out_dim, 1),
        )

    def forward(self, feature):
        f = feature[-1]
        pool_shape = f.shape[2:]

        ppm_out = [f]
        for pool, conv in zip(self.ppm_pooling, self.ppm_conv):
            p = pool(f)
            p = F.interpolate(p, size=pool_shape, mode='bilinear', align_corners=False)
            p = conv(p)
            ppm_out.append(p)
        ppm_out = torch.cat(ppm_out, 1)
        down = self.ppm_out(ppm_out)

        # --------------------------------------
        fpn_out = [down]
        for i in reversed(range(len(feature) - 1)):
            lateral = feature[i]
            lateral = self.fpn_in[i](lateral)  # lateral branch
            down = F.interpolate(down, size=lateral.shape[2:], mode='bilinear', align_corners=False)  # top-down branch
            down = down + lateral
            fpn_out.append(self.fpn_out[i](down))

        fpn_out.reverse()  # [P2 - P5]
        fusion_shape = fpn_out[0].shape[2:]
        fusion = [fpn_out[0]]
        for i in range(1, len(fpn_out)):
            fusion.append(
                F.interpolate(fpn_out[i], fusion_shape, mode='bilinear', align_corners=False)
            )
        x = self.fpn_fuse(torch.cat(fusion, 1))

        return x, fusion


pretrain_dir = '/root/.cache/torch/hub/checkpoints'
if 1:
    cfg = dict(

        # configs/_base_/models/upernet_swin.py
        basic=dict(
            swin=dict(
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.3,
                ape=False,
                patch_norm=True,
                out_indices=(0, 1, 2, 3),
                use_checkpoint=False
            ),

        ),

        # configs/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k.py
        swin_tiny_patch4_window7_224=dict(
            checkpoint=pretrain_dir + '/swin_tiny_patch4_window7_224_22k.pth',

            swin=dict(
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                ape=False,
                drop_path_rate=0.3,
                patch_norm=True,
                use_checkpoint=False,
            ),
            upernet=dict(
                in_channels=[96, 192, 384, 768],
            ),
        ),

        # /configs/swin/upernet_swin_small_patch4_window7_512x512_160k_ade20k.py
        swin_small_patch4_window7_224_22k=dict(
            checkpoint=pretrain_dir + '/swin_small_patch4_window7_224_22k.pth',

            swin=dict(
                embed_dim=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                ape=False,
                drop_path_rate=0.3,
                patch_norm=True,
                use_checkpoint=False
            ),
            upernet=dict(
                in_channels=[96, 192, 384, 768],
            ),
        ),
    )


class Net1(nn.Module):

    def load_pretrain(self, ):

        checkpoint = cfg[self.arch]['checkpoint']
        print('loading %s ...' % checkpoint)
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)['model']
        # if 1:
        #     skip = ['relative_coords_table', 'relative_position_index']
        #     filtered = {}
        #     for k, v in checkpoint.items():
        #         if any([s in k for s in skip]): continue
        #         filtered[k] = v
        #     checkpoint = filtered
        print(self.encoder.load_state_dict(checkpoint, strict=False))  # True

    def __init__(self, ):
        super(Net1, self).__init__()
        self.output_type = ['inference', 'loss']

        self.rgb = RGB()
        self.arch = 'swin_small_patch4_window7_224_22k'

        self.encoder = SwinTransformerV1(
            **{**cfg['basic']['swin'], **cfg[self.arch]['swin'],
               **{'out_norm': LayerNorm2d}}
        )
        encoder_dim = cfg[self.arch]['upernet']['in_channels']
        # [96, 192, 384, 768]

        self.decoder = UPerDecoder(
            in_dim=encoder_dim,
            ppm_pool_scale=[1, 2, 3, 6],
            ppm_dim=512,
            fpn_out_dim=256
        )

        self.logit = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1)
        )
        self.aux = nn.ModuleList([
            nn.Conv2d(256, 1, kernel_size=1, padding=0) for i in range(4)
        ])

    def forward(self, batch):

        # x = batch['image']
        x = batch['image']
        B, C, H, W = x.shape
        x = self.rgb(x)
        encoder = self.encoder(x)
        # print([f.shape for f in encoder])

        # ---
        last, decoder = self.decoder(encoder)
        # print([f.shape for f in decoder])
        # print(last.shape)

        # ---
        logit = self.logit(last)
        logit = F.interpolate(logit, size=None, scale_factor=4, mode='bilinear', align_corners=False)
        # print(logit.shape)

        # ---

        output = {}
        if 'loss' in self.output_type:
            output['bce_loss'] = F.binary_cross_entropy_with_logits(logit, batch['mask'])
            for i in range(4):
                output['aux%d_loss' % i] = criterion_aux_loss(self.aux[i](decoder[i]), batch['mask'])

        if 'inference' in self.output_type:
            output['probability'] = torch.sigmoid(logit)

        return output


models_seg = []
model_seg_nolung = []

for path in MODELS3:
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model = Net()
    model.load_state_dict(state_dict)
    model.float()
    model.eval()
    model.to(device)
    models_seg.append(model)

for path in MODELS4:
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model = Net1()
    model.load_state_dict(state_dict)
    model.float()
    model.eval()
    model.to(device)
    models_seg.append(model)

for path in MODELS_nolung_3:
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model = Net()
    model.load_state_dict(state_dict)
    model.float()
    model.eval()
    model.to(device)
    model_seg_nolung.append(model)

for path in MODELS_nolung_4:
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model = Net1()
    model.load_state_dict(state_dict)
    model.float()
    model.eval()
    model.to(device)
    model_seg_nolung.append(model)


del state_dict


names, preds = [], []
with open(img_txt, 'w') as imgtxt, open(mask_txt, 'w') as masktxt:
    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        idx = str(row['id'])
        pixel_size = row['pixel_size']
        img_height = row['img_height']
        organ = row['organ']
        source = row['data_source']

        mask_segformer = 0
        resizeshape_center = math.ceil((3000 * pixel_size / 1.6) / 32) * 32
        resizeshape_low = int(resizeshape_center * 0.8 / 32) * 32
        resizeshape_high = int(resizeshape_center * 1.2 / 32) * 32
        for resizeshape in [resizeshape_center, resizeshape_low, resizeshape_high]:
            ds = HuBMAPDatasetseg(idx, resizeshape=resizeshape)
            dl = DataLoader(ds, bs, num_workers=0, shuffle=False, pin_memory=True)
            if organ == 'prostate':
                mp = Model_predseg(models_seg, dl, ds.shape[1])
            else:
                mp = Model_predseg(model_seg_nolung, dl, ds.shape[1])
            mask = torch.zeros(len(ds), ds.shape[1], ds.shape[1], dtype=torch.float32)
            for p, i in iter(mp): mask[i.item()] = p.squeeze(-1)
            mask_segformer += mask.view(1, 1, ds.shape[1], ds.shape[1]).permute(0, 2, 1, 3).reshape(1 * ds.shape[1], 1 * ds.shape[1])
        mask_segformer = mask_segformer / 3.

        mask_combine = mask_segformer
        pred = torch.where(mask_combine > 0.4 * len(models_seg), 1, 0)
        maskmask = pred.numpy() * 255
        
        if maskmask.max() == 0:
            print("Empty mask!")
            continue
            
        maskmask = maskmask.astype(np.uint8)
        mask_1 = Image.fromarray(maskmask)
        mask_1.save(os.path.join(mask_dir, idx), 'png')

        imgtxt.write(os.path.join(DATA, idx) + '\n')
        masktxt.write(os.path.join(mask_dir, idx) + '\n')

