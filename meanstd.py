import glob
from PIL import Image
from tqdm import tqdm
import numpy as np

train_path = "/root/autodl-tmp/data_cut_480_x3_fold_1/*.png"
max_pixel_value = 255
sums = []
for img_path in tqdm(glob.glob(train_path)):
    pil_img = Image.open(img_path)
    array_img = np.array(pil_img)
    normalized_array_img = array_img / max_pixel_value
    # We compute the sum over width and height dimensions
    current_sum = normalized_array_img.sum(axis=(0, 1))
    sums.append(current_sum)

img_size = pil_img.size
number_imgs = len(sums)
count = img_size[0] * img_size[1] * number_imgs

computed_mean = np.array(sums).sum(axis=0) / count

squared_centered_sums = []
for img_path in tqdm(glob.glob(train_path)):
    pil_img = Image.open(img_path)
    squared_centered_current_sum = (((np.array(pil_img) / max_pixel_value) - computed_mean) ** 2).sum(axis=(0, 1))
    squared_centered_sums.append(squared_centered_current_sum)

computed_std = (np.array(squared_centered_sums).sum(axis=0) / (count)) ** 0.5
print(computed_mean, computed_std)