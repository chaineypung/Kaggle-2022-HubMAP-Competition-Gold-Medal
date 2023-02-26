import numpy as np
import pandas as pd
import os
import rasterio
from torch.utils.data import Dataset
from rasterio.windows import Window
import cv2
from tqdm import tqdm
from PIL import Image
import tifffile as tiff

MASKS = r'/root/autodl-tmp/train.csv'
DATA = r'/root/autodl-tmp/train_images'
MASK = r'/root/autodl-tmp/mask'

path = r'/root/autodl-tmp'
DATA_cut = r'/root/autodl-tmp/data_cut_256_x2_fold_2'
MASK_cut = r'/root/autodl-tmp/mask_cut_256_x2_fold_2'

img_txt = r'./train_valid_list_path/train_img_fold_2.txt'
mask_txt = r'./train_valid_list_path/train_mask_fold_2.txt'

if not os.path.exists(DATA_cut):
    os.mkdir(DATA_cut)
    os.mkdir(MASK_cut)

f = open(img_txt, 'r')
img_list = list(f)
f.close()

f = open(mask_txt, 'r')
mask_list = list(f)
f.close() 

K = 2
sz = 256
reduce = 2
overlap = 256
s_th = 0
p_th = 1000 * (sz // 256) ** 2
mask_th = 2000

df_masks = pd.read_csv(MASKS)[['id', 'rle']].set_index('id')
df_masks.head()
df = pd.read_csv(os.path.join(MASKS))

class HuBMAPDataset(Dataset):

    def __init__(self, img_idx, mask_idx, sz=sz, reduce=reduce, overlap=overlap):
        self.data = rasterio.open(img_idx, num_threads='all_cpus')
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        self.shape = self.data.shape
        self.reduce = reduce
        self.sz = reduce * sz
        self.step = overlap
        self.pad0 = (self.sz - self.shape[0] % self.step) % self.step
        self.pad1 = (self.sz - self.shape[1] % self.step) % self.step
        self.n0max = (self.shape[0] + self.pad0 - self.sz) // self.step + 1
        self.n1max = (self.shape[1] + self.pad1 - self.sz) // self.step + 1
        self.mask = np.asarray(Image.open(mask_idx))

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        n0, n1 = idx // self.n1max, idx % self.n1max
        x0, y0 = -self.pad0 // 2 + n0 * self.step, -self.pad1 // 2 + n1 * self.step
        p00, p01 = max(0, x0), min(x0 + self.sz, self.shape[0])
        p10, p11 = max(0, y0), min(y0 + self.sz, self.shape[1])

        if p01 - p00 != self.sz or p11 - p10 != self.sz:
            if p01 - p00 != self.sz:
                if p00 == 0:
                    p01 = p01 + self.sz - (p01 - p00)
                else:
                    p00 = p00 - (self.sz - (p01 - p00))
            if p11 - p10 != self.sz:
                if p10 == 0:
                    p11 = p11 + self.sz - (p11 - p10)
                else:
                    p10 = p10 - (self.sz - (p11 - p10))
            img = np.moveaxis(self.data.read([1, 2, 3], window=Window.from_slices((p00, p01), (p10, p11))), 0, -1)
            mask = self.mask[p00:p01, p10:p11]
        else:
            img = np.moveaxis(self.data.read([1, 2, 3], window=Window.from_slices((p00, p01), (p10, p11))), 0, -1)
            mask = self.mask[p00:p01, p10:p11]

        if self.reduce != 1:
            img = cv2.resize(img, (self.sz // reduce, self.sz // reduce),
                             interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.sz // reduce, self.sz // reduce),
                              interpolation=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        return img, mask, (-1 if (s > s_th).sum() <= p_th or img.sum() <= p_th or (mask > 0).sum() <= mask_th else idx)


x_tot, x2_tot = [], []
with open(os.path.join(path, f"train_img_256_x2_fold_{K}.txt"), 'w') as train, open(os.path.join(path, f"train_mask_256_x2_fold_{K}.txt"), 'w') as test:
    for img_index, mask_index in tqdm(zip(img_list, mask_list)):
        img_index = img_index.strip('\n')
        mask_index = mask_index.strip('\n')
        index = int(img_index.split("images/")[1].split(".tiff")[0])
        ds = HuBMAPDataset(img_idx=img_index, mask_idx=mask_index)
        for i in range(len(ds)):
            im, m, idx = ds[i]
            if idx < 0: continue
            im = Image.fromarray(im)
            im.save(os.path.join(DATA_cut, f'{index}_{idx:04d}.png'), 'png')
            m = Image.fromarray(m)
            m.save(os.path.join(MASK_cut, f'{index}_{idx:04d}.png'), 'png')
            train.write(os.path.join(DATA_cut, f'{index}_{idx:04d}.png') + '\n')
            test.write(os.path.join(MASK_cut, f'{index}_{idx:04d}.png') + '\n')
