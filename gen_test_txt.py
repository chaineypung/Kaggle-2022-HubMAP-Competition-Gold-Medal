import csv
import os
import cv2
from tqdm import tqdm
import pandas as pd
import random

csv_path = r'/root/autodl-tmp/Warwick_colon/extrain.csv'
image_path = r'/root/autodl-tmp/Warwick_colon/train'

f = open(csv_path, 'w', encoding='utf-8', newline = '')
csv_writer = csv.writer(f)
csv_writer.writerow(["id", "pixel_size", "img_height", "organ", "data_source"])
for image in os.listdir(image_path):
    image_file_path = image_path + '/' + image
    img_tmp = cv2.imread(image_file_path, 0)
    img_height = img_tmp.shape[0]
    csv_writer.writerow([image, 0.4, img_height, "colon", "Hubmap"])
    print(f"{image} done!")
f.close()