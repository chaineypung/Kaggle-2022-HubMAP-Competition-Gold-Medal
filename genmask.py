# import os
# import numpy as np
# import pandas as pd
# import tifffile as tiff
# from PIL import Image
# from tqdm import tqdm
# import rasterio

# data_dir = r'/root/autodl-tmp/kidney/train'
# mask_dir = r'/root/autodl-tmp/kidney/mask'
# MASKS = r'/root/autodl-tmp/kidney/train.csv'

# if not os.path.exists(mask_dir):
#     os.mkdir(mask_dir)

# def mask2rle(img):
#     '''
#     img: numpy array, 1 - mask, 0 - background
#     Returns run length as string formated
#     '''
#     pixels = img.T.flatten()
#     pixels = np.concatenate([[0], pixels, [0]])
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
#     runs[1::2] -= runs[::2]
#     return ' '.join(str(x) for x in runs)


# def rle2mask(mask_rle, shape=(1600, 256)):
#     '''
#     mask_rle: run-length as string formated (start length)
#     shape: (width,height) of array to return
#     Returns numpy array, 1 - mask, 0 - background

#     '''
#     s = mask_rle.split()
#     starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
#     starts -= 1
#     ends = starts + lengths
#     img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
#     for lo, hi in zip(starts, ends):
#         img[lo:hi] = 1
#     return img.reshape(shape).T


# df = pd.read_csv(os.path.join(MASKS))
# for img_name in tqdm(os.listdir(data_dir)):
#     img_1 = tiff.imread(os.path.join(data_dir, img_name))
#     img_1 = np.moveaxis(img_1.read([1, 2, 3]), 0, -1)
#     print(img_1.shape[0], img_1.shape[1])
#     img__name = img_name.split('.')[0]
#     mask_1 = rle2mask(df[df["id"] == img__name]["encoding"].iloc[-1], (img_1.shape[1], img_1.shape[0])) * 255
#     mask_1 = Image.fromarray(mask_1)
#     mask_1.save(os.path.join(mask_dir, img_name.split('.')[0] + '.png'), 'png')




import os
import numpy as np
import pandas as pd
import tifffile as tiff
from PIL import Image
from tqdm import tqdm
import rasterio

data_dir = r'/root/autodl-tmp/kidney/train'
mask_dir = r'/root/autodl-tmp/kidney/mask'
MASKS = r'/root/autodl-tmp/kidney/train.csv'

if not os.path.exists(mask_dir):
    os.mkdir(mask_dir)

def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle, shape=(1600, 256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


df = pd.read_csv(os.path.join(MASKS))
for img_name in tqdm(os.listdir(data_dir)):
    img_1 = rasterio.open(os.path.join(data_dir, img_name), transform=rasterio.Affine(1, 0, 0, 0, 1, 0), num_threads='all_cpus')
    img__name = img_name.split('.')[0]
    mask_1 = rle2mask(df[df["id"] == img__name]["encoding"].iloc[-1], (img_1.shape[1], img_1.shape[0])) * 255
    mask_1 = Image.fromarray(mask_1)
    mask_1.save(os.path.join(mask_dir, img_name.split('.')[0] + '.png'), 'png')
