import numpy as np
import pandas as pd
import os
import rasterio
from torch.utils.data import Dataset
from rasterio.windows import Window
import cv2
from tqdm import tqdm
from PIL import Image
import tifffile
from tqdm.contrib import tzip
from scipy.ndimage.morphology import distance_transform_edt


def read_tiff(path, scale=None, verbose=0): 
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


def onehot_to_multiclass_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to an edgemap (K,H,W)
    """
    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    channels = []
    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        dist = (dist > 0).astype(np.uint8)
        channels.append(dist)

    return np.array(channels)


def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)
    """

    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])
    for i in range(num_classes):
        # ti qu lun kuo
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    # edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)*255
    return edgemap


def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector
    """
    _mask = [mask == (i) for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)


class HuBMAPDataset(Dataset):

    def __init__(self, img_idx, mask_idx, organ):
        self.img = cv2.cvtColor(read_tiff(img_idx), cv2.COLOR_RGB2BGR)
        self.mask = cv2.imread(mask_idx, cv2.IMREAD_GRAYSCALE)
        self.organ = organ

    def __len__(self):
        return 1

    def __getitem__(self, idx):
#         if self.organ == 'prostate':
#             self.img = cv2.resize(self.img, (190, 190),
#                        interpolation=cv2.INTER_AREA)
#         self.img = cv2.resize(self.img, (190, 190),
#                          interpolation=cv2.INTER_AREA)
#         self.mask = cv2.resize(self.mask, (190, 190),
#                              interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(self.img, (img_size, img_size),
                         interpolation=cv2.INTER_AREA)
        mask = cv2.resize(self.mask, (mask_size, mask_size),
                          interpolation=cv2.INTER_NEAREST)
        return img, mask, 1
    
    
for K in [1, 2, 3, 4, 5]:
    for mode in ['train', 'valid']:
        MASKS = r'/root/autodl-tmp/train.csv'
        DATA = r'/root/autodl-tmp/train_images'
        MASK = r'/root/autodl-tmp/mask'

        path = r'/root/autodl-tmp'
        DATA_cut = f'/root/autodl-tmp/{mode}_data_cut_768_x3_fold_{K}_lung'
        MASK_cut = f'/root/autodl-tmp/{mode}_mask_cut_768_x3_fold_{K}_lung'
        EDGE_cut = f'/root/autodl-tmp/{mode}_edge_cut_768_x3_fold_{K}_lung'

        img_txt = f'./train_valid_list_path/{mode}_img_fold_{K}.txt'
        mask_txt = f'./train_valid_list_path/{mode}_mask_fold_{K}.txt'

        data_cut_txt = f"{mode}_img_768_x3_fold_{K}_lung.txt"
        mask_cut_txt = f"{mode}_mask_768_x3_fold_{K}_lung.txt"
        edge_cut_txt = f"{mode}_edge_768_x3_fold_{K}_lung.txt"

        if mode == 'train':
            img_size = 768
            mask_size = 768

        if mode == 'valid':
            img_size = 768
            mask_size = 768

        if not os.path.exists(EDGE_cut):
            os.mkdir(EDGE_cut)

        if not os.path.exists(DATA_cut):
            os.mkdir(DATA_cut)
            os.mkdir(MASK_cut)

        f = open(img_txt, 'r')
        img_list = list(f)
        f.close()

        f = open(mask_txt, 'r')
        mask_list = list(f)
        f.close()

        df_masks = pd.read_csv(MASKS)[['id', 'rle']].set_index('id')
        df_masks.head()
        df = pd.read_csv(os.path.join(MASKS))

        if mode == 'valid':
            with open(os.path.join(path, data_cut_txt), 'w') as train, open(os.path.join(path, mask_cut_txt), 'w') as test:
                for img_index, mask_index in tzip(img_list, mask_list):
                    img_index = img_index.strip('\n')
                    mask_index = mask_index.strip('\n')
                    index = int(img_index.split("images/")[1].split(".tiff")[0])
                    organ = df[df["id"] == index]["organ"].iloc[-1]
                    if organ != 'lung':
                        continue
                    ds = HuBMAPDataset(img_idx=img_index, mask_idx=mask_index, organ=organ)
                    for i in range(len(ds)):
                        im, m, idx = ds[i]
                        if idx < 0: continue
                        cv2.imwrite(os.path.join(DATA_cut, f'{index}_{idx:04d}.png'), im)
                        cv2.imwrite(os.path.join(MASK_cut, f'{index}_{idx:04d}.png'), m)
                        train.write(os.path.join(DATA_cut, f'{index}_{idx:04d}.png') + '\n')
                        test.write(os.path.join(MASK_cut, f'{index}_{idx:04d}.png') + '\n')

        if mode == 'train':
            with open(os.path.join(path, data_cut_txt), 'w') as train, open(os.path.join(path, mask_cut_txt), 'w') as test, open(os.path.join(path, edge_cut_txt), 'w') as edge:
                for img_index, mask_index in tzip(img_list, mask_list):
                    img_index = img_index.strip('\n')
                    mask_index = mask_index.strip('\n')
                    index = int(img_index.split("images/")[1].split(".tiff")[0])
                    organ = df[df["id"] == index]["organ"].iloc[-1]
                    if organ != 'lung':
                        continue
                    ds = HuBMAPDataset(img_idx=img_index, mask_idx=mask_index, organ=organ)
                    for i in range(len(ds)):
                        im, m, idx = ds[i]
                        if idx < 0: continue
                        oneHot_label = mask_to_onehot(m, 2)
                        edgee = onehot_to_binary_edges(oneHot_label, 2, 2) 
                        edgee[:2, :] = 0
                        edgee[-2:, :] = 0
                        edgee[:, :2] = 0
                        edgee[:, -2:] = 0
                        cv2.imwrite(os.path.join(DATA_cut, f'{index}_{idx:04d}.png'), im)
                        cv2.imwrite(os.path.join(MASK_cut, f'{index}_{idx:04d}.png'), m)
                        cv2.imwrite(os.path.join(EDGE_cut, f'{index}_{idx:04d}.png'), edgee)
                        train.write(os.path.join(DATA_cut, f'{index}_{idx:04d}.png') + '\n')
                        test.write(os.path.join(MASK_cut, f'{index}_{idx:04d}.png') + '\n')
                        edge.write(os.path.join(EDGE_cut, f'{index}_{idx:04d}.png') + '\n')