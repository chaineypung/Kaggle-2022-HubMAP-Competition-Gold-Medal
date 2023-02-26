import cv2
from tqdm import tqdm
import pandas as pd
from util import *

csv_path = r'/root/autodl-tmp/spleen/train1.csv'
image_path = r'/root/autodl-tmp/spleen/mask1'
info_path = r'/root/autodl-tmp/train.csv'
data_dir = r'/root/autodl-tmp/spleen/img'
train_valid_list_path = r'./train_valid_list_path_newspleen'

fold = 5

df = pd.read_csv(os.path.join(info_path))

prostate = 0
spleen = 0
lung = 0
kidney = 0
largeintestine = 0

# organ_index = {'prostate': 0, 'spleen': 1, 'lung': 2, 'kidney': 3, 'largeintestine': 4}
# f = open(csv_path, 'w', encoding='utf-8', newline = '')
# csv_writer = csv.writer(f)
# csv_writer.writerow(["ID", "CATE", "size"])
# for image in tqdm(os.listdir(image_path)):
#     img_name = int(image.split('.')[0].split('_')[0])
#     image_file_path = image_path + '/' + image
#     img_tmp = cv2.imread(image_file_path, 0)
#     size = (img_tmp == 255).sum()
#     p = organ_index[df[df["id"] == img_name]["organ"].iloc[-1]]
#     if p == 0:
#         prostate += 1
#     if p == 1:
#         spleen += 1
#     if p == 2:
#         lung += 1
#     if p == 3:
#         kidney += 1
#     if p == 4:
#         largeintestine += 1
#     csv_writer.writerow([image, p, size])
# f.close()
# print(prostate, spleen, lung, kidney, largeintestine)

for K in range(fold):
    train, valid = get_fold_filelist(csv_path, fold, K)
    train_list = [data_dir + sep + i[0] for i in train]
    train_list_GT = [image_path + sep + i[0] for i in train]
    valid_list = [data_dir + sep + i[0] for i in valid]
    valid_list_GT = [image_path + sep + i[0] for i in valid]
#     for i in range(len(train_list)):
#         train_list[i] = train_list[i].replace('png', 'tiff')
#     for i in range(len(valid_list)):
#         valid_list[i] = valid_list[i].replace('png', 'tiff')
    f = open(os.path.join(train_valid_list_path, f"train_img_fold_{K + 1}.txt"),"w")
    for line in train_list:
        f.write(line + '\n')
    f.close()
    f = open(os.path.join(train_valid_list_path, f"train_mask_fold_{K + 1}.txt"), "w")
    for line in train_list_GT:
        f.write(line + '\n')
    f.close()
    f = open(os.path.join(train_valid_list_path, f"valid_img_fold_{K + 1}.txt"), "w")
    for line in valid_list:
        f.write(line + '\n')
    f.close()
    f = open(os.path.join(train_valid_list_path, f"valid_mask_fold_{K + 1}.txt"), "w")
    for line in valid_list_GT:
        f.write(line + '\n')
    f.close()






