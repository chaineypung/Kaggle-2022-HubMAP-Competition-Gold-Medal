import numpy as np
import pandas as pd
import os
import rasterio
from torch.utils.data import Dataset
from rasterio.windows import Window
import cv2
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


MASKS = r'/root/autodl-tmp/Warwick_colon/extrain.csv'
DATA = r'/root/autodl-tmp/Warwick_colon/train'
MASK = r'/root/autodl-tmp/Warwick_colon/train_mask'

DATA_cut = r'/root/autodl-tmp/Warwick_colon/data_cut'
MASK_cut = r'/root/autodl-tmp/Warwick_colon/mask_cut'

os.mkdir(DATA_cut)
os.mkdir(MASK_cut)

sz = 768   # the size of tiles
reduce = 4 # reduce the original images by 4 times
overlap = 2800

s_th = 40  # saturation blancking threshold
p_th = 1000 * (sz // 256) ** 2  # threshold for the minimum number of pixels
mask_th = 5000

df_masks = pd.read_csv(MASKS)[['id', 'pixel_size']].set_index('id')
df_masks.head()

class HuBMAPDataset(Dataset):

    def __init__(self, idx, sz=sz, reduce=reduce, overlap = overlap, encs=None):
        self.data = rasterio.open(os.path.join(DATA, str(idx)), num_threads='all_cpus')
        # some images have issues with their format
        # and must be saved correctly before reading with rasterio
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
        # self.mask = np.asarray(Image.open(os.path.join(MASK, str(idx) + '.png')))
        self.mask = rasterio.open(os.path.join(MASK, str(idx)), num_threads='all_cpus')
        self.mask = self.mask.read([1])[0]
        

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

for index, encs in tqdm(df_masks.iterrows(), total=len(df_masks)):
    ds = HuBMAPDataset(index, encs=encs)
    for i in range(len(ds)):
        im, m, idx = ds[i]
        if idx < 0: continue
        im = Image.fromarray(im)
        im.save(os.path.join(DATA_cut, f'{index}_{idx:04d}.png'), 'png')
        m = Image.fromarray(m)
        m.save(os.path.join(MASK_cut, f'{index}_{idx:04d}.png'), 'png')


