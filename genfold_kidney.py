import csv
import os
import cv2
from tqdm import tqdm
import pandas as pd
import random

csv_path = r'/root/autodl-tmp/spleen/train1.csv'
image_path = r'/root/autodl-tmp/spleen/mask1'

f = open(csv_path, 'w', encoding='utf-8', newline = '')
csv_writer = csv.writer(f)
csv_writer.writerow(["ID", "CATE", "size"])
for image in os.listdir(image_path):
    image_file_path = image_path + '/' + image
    img_tmp = cv2.imread(image_file_path, 0)
    print(img_tmp.max())
    size = (img_tmp > 0).sum()
    p = random.randint(0, 1)
    csv_writer.writerow([image, p, size])
    print(f"{image} done!")
f.close()
