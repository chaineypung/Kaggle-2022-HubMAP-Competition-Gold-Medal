# # import torchvision.transforms as transforms
# from torch.utils.data import Dataset
# from PIL import Image
# import os
# from albumentations import *
# import torch
# import cv2
# import numpy as np
# import random

# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])

# # mean = np.array([0.7720342, 0.74582646, 0.76392896])
# # std = np.array([0.24745085, 0.26182273, 0.25782376])

# # mean = np.array([0.79117177, 0.7583153, 0.78292631])
# # std = np.array([[0.16705585, 0.19441158, 0.18427182]])

# def do_random_crop(image, mask, size):
#     height, width = image.shape[:2]
#     x = np.random.choice(width -size) if width>size else 0
#     y = np.random.choice(height-size) if height>size else 0
#     image = image[y:y+size,x:x+size]
#     mask  = mask[y:y+size,x:x+size]
#     return image, mask

# def img2tensor(img, dtype: np.dtype = np.float32):
#     if img.ndim == 2: img = np.expand_dims(img, 2)
#     img = np.transpose(img, (2, 0, 1))
#     return torch.from_numpy(img.astype(dtype, copy=False))

# def get_aug(p=1.0):
#     return Compose([
#         HorizontalFlip(p=0.5),
#         VerticalFlip(),
#         RandomRotate90(p=1),
#         #Morphology
#         ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2,0.2), rotate_limit=(-30,30),
#                          interpolation=1, border_mode=0, value=(0,0,0), p=0.5),
#         GaussNoise(var_limit=(0,50.0), mean=0, p=0.5),
#         GaussianBlur(blur_limit=(3,7), p=0.5),
#         #Color
#         RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5,
#                                  brightness_by_max=True,p=0.5),
#         HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30,
#                            val_shift_limit=0, p=0.5),
#         OneOf([
#             OpticalDistortion(p=0.3),
#             GridDistortion(p=.1),
#             PiecewiseAffine(p=0.3),
#         ], p=0.3),
#     ], p=p)


# def get_aug_enhance(p=1.0):
#     return Compose([
#         HorizontalFlip(p=0.5),
#         VerticalFlip(),
#         RandomRotate90(p=1),
#         #Morphology
#         ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2, 0.2), rotate_limit=(-30, 30),
#                          interpolation=1, border_mode=0, value=(0, 0, 0), p=0.5),
#         GaussNoise(var_limit=(0, 50.0), mean=0, p=0.5),
#         GaussianBlur(blur_limit=(3, 7), p=0.5),
#         #Color
#         RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5,
#                                  brightness_by_max=True, p=0.5),
#         RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
#         HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30,
#                            val_shift_limit=10, p=0.5),
#         OneOf([
#             OpticalDistortion(p=0.3),
#             GridDistortion(p=.1),
#             PiecewiseAffine(p=0.3),
#         ], p=0.4),
#     ], p=p)


# def get_aug_prostate(p=1.0):
#     return Compose([
#         HorizontalFlip(p=0.5),
#         VerticalFlip(),
#         RandomRotate90(p=1),
#         #Morphology
#         ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2, 0.2), rotate_limit=(-30, 30),
#                          interpolation=1, border_mode=0, value=(0, 0, 0), p=0.5),
#         GaussNoise(var_limit=(0, 50.0), mean=0, p=0.5),
#         GaussianBlur(blur_limit=(3, 7), p=0.5),
#         #Color
#         RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5,
#                                  brightness_by_max=True, p=0.5),
#         RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
#         HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30,
#                            val_shift_limit=10, p=0.5),
#         OneOf([
#             OpticalDistortion(p=0.3),
#             GridDistortion(p=.1),
#             PiecewiseAffine(p=0.3),
#         ], p=0.4),
#     ], p=p)


# class Train_Dataset(Dataset):

#     def __init__(self, img_list, label_list, transform=None, target_transform=None, image_resolution=None, tmf=get_aug_enhance()):

#         self.img_dir = img_list
#         self.label_dir = label_list

#         self.transform = transform
#         self.target_transform = target_transform
#         self.image_resolution = image_resolution
#         self.tmf = tmf

#     def __getitem__(self, idx):

#         image_name = self.img_dir[idx]
#         label_name = self.label_dir[idx]

#         img = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
#         label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)

#         img = np.array(img)
#         label = np.array(label)
        
# #         crop_size = random.randint(256, 512)
# #         img, label = do_random_crop(image, label, crop_size)
# #         img = Resize(img, 512, 512, interpolation=cv2.INTER_AREA)
# #         label = Resize(label, 512, 512, interpolation=cv2.INTER_NEAREST)
        
#         augmented = self.tmf(image=img, mask=label)
#         img, label = augmented['image'], augmented['mask']

#         return img2tensor((img / 255.0 - mean) / std), img2tensor(label / 255.)


#     def __len__(self):
#         return len(self.img_dir)


# class Valid_Dataset(Dataset):

#     def __init__(self, img_list, label_list, transform=None, target_transform=None, image_resolution=None):

#         self.img_dir = img_list
#         self.label_dir = label_list
#         self.transform = transform
#         self.target_transform = target_transform
#         self.image_resolution = image_resolution

#     def __getitem__(self, idx):

#         image_name = self.img_dir[idx]
#         label_name = self.label_dir[idx]

#         img = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
#         label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)

#         img = np.array(img)
#         label = np.array(label)

#         return img2tensor((img / 255.0 - mean) / std), img2tensor(label / 255.)

#     def __len__(self):
#         return len(self.img_dir)


# class Train_Dataset_cls(Dataset):

#     def __init__(self, img_list, label_list, transform=None, target_transform=None, image_resolution=None, df=None, tmf=get_aug_enhance(), tmf_p=get_aug_prostate()):

#         self.img_dir = img_list
#         self.label_dir = label_list

#         self.transform = transform
#         self.target_transform = target_transform
#         self.image_resolution = image_resolution
#         self.tmf = tmf
#         self.tmf_p = tmf_p
#         self.df = df

#     def __getitem__(self, idx):

#         image_name = self.img_dir[idx]
#         label_name = self.label_dir[idx]
        
#         img = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
#         label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)

#         img = np.array(img)
#         label = np.array(label)

#         id_name = image_name.split("fold_1/")[1].split("_")[0] + '.png'
#         cls = np.array(self.df[self.df["ID"] == id_name]["CATE"].iloc[-1])
        
#         if cls == 0:
#             augmented = self.tmf_p(image=img, mask=label)
#             img, label = augmented['image'], augmented['mask']
#         else:
#             augmented = self.tmf(image=img, mask=label)
#             img, label = augmented['image'], augmented['mask']

#         return img2tensor((img / 255.0 - mean) / std), img2tensor(label / 255.), cls


#     def __len__(self):
#         return len(self.img_dir)


# class Valid_Dataset_cls(Dataset):

#     def __init__(self, img_list, label_list, transform=None, target_transform=None, image_resolution=None, df=None):

#         self.img_dir = img_list
#         self.label_dir = label_list
#         self.transform = transform
#         self.target_transform = target_transform
#         self.image_resolution = image_resolution
#         self.df = df

#     def __getitem__(self, idx):

#         image_name = self.img_dir[idx]
#         label_name = self.label_dir[idx]

#         id_name = image_name.split("256/")[1]
#         cls = torch.from_numpy(np.array(self.df[self.df["ID"] == id_name]["CATE"].iloc[-1]))

#         img = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
#         label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)

#         img = np.array(img)
#         label = np.array(label)

#         return img2tensor((img / 255.0 - mean) / std), img2tensor(label / 255.), cls

#     def __len__(self):
#         return len(self.img_dir)










# import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os
from albumentations import *
import torch
import cv2
import numpy as np
import random
import pandas as pd

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# mean = np.array([0.7720342, 0.74582646, 0.76392896])
# std = np.array([0.24745085, 0.26182273, 0.25782376])

def do_random_flip(image, mask, edge):
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
        edge = cv2.flip(edge, 0)
    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        edge = cv2.flip(edge, 1)
    if np.random.rand() > 0.5:
        image = image.transpose(1, 0, 2)
        mask = mask.transpose(1, 0)
        edge = edge.transpose(1, 0)

    image = np.ascontiguousarray(image)
    mask = np.ascontiguousarray(mask)
    edge = np.ascontiguousarray(edge)
    return image, mask, edge


def do_random_rot90(image, mask, edge):
    r = np.random.choice([
        0,
        cv2.ROTATE_90_CLOCKWISE,
        cv2.ROTATE_90_COUNTERCLOCKWISE,
        cv2.ROTATE_180,
    ])
    if r == 0:
        return image, mask, edge
    else:
        image = cv2.rotate(image, r)
        mask = cv2.rotate(mask, r)
        edge = cv2.rotate(edge, r)
        return image, mask, edge


# crop ##----
def do_crop(image, mask, size, xy=(0, 0)):
    height, width = image.shape[:2]
    x, y = xy
    if x is None: x = (width - size) // 2
    if y is None: y = (height - size) // 2

    image = image[y:y + size, x:x + size]
    mask = mask[y:y + size, x:x + size]
    return image, mask


def do_random_crop(image, mask, size):
    height, width = image.shape[:2]
    x = np.random.choice(width - size) if width > size else 0
    y = np.random.choice(height - size) if height > size else 0
    image = image[y:y + size, x:x + size]
    mask = mask[y:y + size, x:x + size]
    return image, mask


# transform ##----
def do_random_rotate_scale(image, mask, edge, angle=30, scale=[0.8, 1.2]):
    angle = np.random.uniform(-angle, angle)
    scale = np.random.uniform(*scale) if scale is not None else 1

    height, width = image.shape[:2]
    center = (height // 2, width // 2)

    transform = cv2.getRotationMatrix2D(center, angle, scale)
    image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpAffine(mask, transform, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    edge = cv2.warpAffine(edge, transform, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask, edge


# noise
def do_random_noise(image, mask, mag=0.1):
    height, width = image.shape[:2]
    noise = np.random.uniform(-1, 1, (height, width, 1)) * mag
    image = image + noise
    image = np.clip(image, 0, 1)
    return image, mask


# https://openreview.net/pdf?id=rkBBChjiG
# <todo> mixup/cutout

# intensity
def do_random_contast(image, mask, mag=0.3):
    alpha = 1 + random.uniform(-1, 1) * mag
    image = image * alpha
    image = np.clip(image, 0, 1)
    return image, mask


def do_random_hsv(image, mask, mag=[0.15,0.25,0.25]):
    image = (image*255).astype(np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0].astype(np.float32)  # hue
    s = hsv[:, :, 1].astype(np.float32)  # saturation
    v = hsv[:, :, 2].astype(np.float32)  # value
    h = (h*(1 + random.uniform(-1,1)*mag[0]))%180
    s =  s*(1 + random.uniform(-1,1)*mag[1])
    v =  v*(1 + random.uniform(-1,1)*mag[2])

    hsv[:, :, 0] = np.clip(h,0,180).astype(np.uint8)
    hsv[:, :, 1] = np.clip(s,0,255).astype(np.uint8)
    hsv[:, :, 2] = np.clip(v,0,255).astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    image = image.astype(np.float32)/255
    return image, mask

# For Kidney
# def do_random_hsv(image, mask, mag=[0.15, 0.25, 0.25]):
#     image = (image * 255).astype(np.uint8)
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     h = hsv[:, :, 0].astype(np.float32)  # hue
#     s = hsv[:, :, 1].astype(np.float32)  # saturation
#     v = hsv[:, :, 2].astype(np.float32)  # value
#     h = (h * (1 + random.uniform(-1, 1) * mag[0])) % 180
#     s = s * (1 + random.uniform(0, mag[1]))
#     v = v * (1 + random.uniform(-mag[2], 0))

#     hsv[:, :, 0] = np.clip(h, 0, 180).astype(np.uint8)
#     hsv[:, :, 1] = np.clip(s, 0, 255).astype(np.uint8)
#     hsv[:, :, 2] = np.clip(v, 0, 255).astype(np.uint8)
#     image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     image = image.astype(np.float32) / 255
#     return image, mask


def do_gray(image, mask):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image, mask


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2: img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))

def image_to_tensor(image, mode='bgr'): #image mode
    if mode=='bgr':
        image = image[:,:,::-1]
    x = image
    x = x.transpose(2,0,1)
    x = np.ascontiguousarray(x)
    x = torch.tensor(x, dtype=torch.float)
    return x

def tensor_to_image(x, mode='bgr'):
    image = x.data.cpu().numpy()
    image = image.transpose(1,2,0)
    if mode=='bgr':
        image = image[:,:,::-1]
    image = np.ascontiguousarray(image)
    image = image.astype(np.float32)
    return image

def mask_to_tensor(mask):
    x = mask
    x = torch.tensor(x, dtype=torch.float)
    x = x.unsqueeze(0)
    return x

def tensor_to_mask(x):
    mask = x.data.cpu().numpy()
    mask = mask.astype(np.float32)
    return mask

def train_augment5b(image, mask, edge, clsy, k):
    image, mask, edge = do_random_flip(image, mask, edge)
    image, mask, edge = do_random_rot90(image, mask, edge)
    
    if clsy == 5:
        for fn in np.random.choice([
            lambda image, mask: (image, mask),
            lambda image, mask: do_random_noise(image, mask, mag=0.1),
            lambda image, mask: do_random_contast(image, mask, mag=0.40),
            lambda image, mask: do_random_hsv(image, mask, mag=[0.30, 0.65, 0.45])
        ], 3): image, mask = fn(image, mask)
    else:
        for fn in np.random.choice([
        lambda image, mask: (image, mask),
        lambda image, mask: do_random_noise(image, mask, mag=0.1),
        lambda image, mask: do_random_contast(image, mask, mag=0.40),
        lambda image, mask: do_random_hsv(image, mask, mag=[0.45, 0.45, 0.1])# 0.0
    ], 2): image, mask = fn(image, mask)
    
    if clsy == 0:
        for fn in np.random.choice([
            lambda image, mask, edge: (image, mask, edge),
            lambda image, mask, edge: do_random_rotate_scale(image, mask, edge, angle=45, scale=[0.7, 1.3]),
        ], 1): image, mask, edge = fn(image, mask, edge)
    else:
        for fn in np.random.choice([
            lambda image, mask, edge: (image, mask, edge),
            lambda image, mask, edge: do_random_rotate_scale(image, mask, edge, angle=45, scale=[0.5, 2.0]),# 0.5, 2.0 # kidney 0.7, 1.3
        ], 1): image, mask, edge = fn(image, mask, edge)

    return image, mask, edge


def get_aug(p=1.0):
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(),
        RandomRotate90(p=1),
        # Morphology
        ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2, 0.2), rotate_limit=(-30, 30),
                         interpolation=1, border_mode=0, value=(0, 0, 0), p=0.5),
        GaussNoise(var_limit=(0, 50.0), mean=0, p=0.5),
        GaussianBlur(blur_limit=(3, 7), p=0.5),
        # Color
        RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5,
                                 brightness_by_max=True, p=0.5),
        HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30,
                           val_shift_limit=0, p=0.5),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            PiecewiseAffine(p=0.3),
        ], p=0.3),
    ], p=p)


def get_aug_enhance(p=1.0):
    return Compose([
        # Color
        # RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.3),
            PiecewiseAffine(p=0.3),
        ], p=0.3),
    ], p=p)


class Train_Dataset(Dataset):

    def __init__(self, img_list, label_list, edge_list, transform=None, target_transform=None, image_resolution=None,
                 tmf=get_aug_enhance(), df=pd.read_csv(os.path.join(r'/root/autodl-tmp/train_cut_256.csv'))):
        self.img_dir = img_list
        self.label_dir = label_list
        self.edge_dir = edge_list

        self.transform = transform
        self.target_transform = target_transform
        self.image_resolution = image_resolution
        self.tmf = tmf
        self.df = df

    def __getitem__(self, idx):
        image_name = self.img_dir[idx]
        label_name = self.label_dir[idx]
        # edge_name = self.edge_dir[idx]
        
        # id_name = image_name.split("all/")[1].split("_")[0] + '.png'
        # clsy = torch.from_numpy(np.array(self.df[self.df["ID"] == id_name]["CATE"].iloc[-1]))
        
        clsy = 0
        
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        label  = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        # edge = cv2.imread(edge_name, cv2.IMREAD_GRAYSCALE)
        
        label[label > 0] = 255
        
        # if image.shape[0] != 256 or image.shape[1] != 256:
        #     image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        #     label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        k = 0
        # if np.random.rand() < 0.5:
        #     image = cv2.resize(image, dsize=None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
        #     image = cv2.resize(image, dsize=(768, 768), interpolation=cv2.INTER_LINEAR)
        #     k = 1

        image = image.astype(np.float32) 
        label = label.astype(np.float32)
        # edge = edge.astype(np.float32) / 255.

        augmented = self.tmf(image=image, mask=label)
        image, label = augmented['image'], augmented['mask']
        
        image = image / 255.
        label = label / 255.

        image, label, edge = train_augment5b(image, label, label, clsy, k)
        
        image = image_to_tensor(image)
        label = mask_to_tensor(label > 0.5)
        edge = mask_to_tensor(edge > 0.5)

        return image, label, edge, clsy

    def __len__(self):
        return len(self.img_dir)


class Valid_Dataset(Dataset):

    def __init__(self, img_list, label_list, transform=None, target_transform=None, image_resolution=None):
        self.img_dir = img_list
        self.label_dir = label_list
        self.transform = transform
        self.target_transform = target_transform
        self.image_resolution = image_resolution

    def __getitem__(self, idx):
        image_name = self.img_dir[idx]
        label_name = self.label_dir[idx]

        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        label  = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        
        label[label > 0] = 255
        
        # if image.shape[0] != 256 or image.shape[1] != 256:
        #     image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        #     label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.
        label = label.astype(np.float32) / 255.
        
        image = image_to_tensor(image)
        label = mask_to_tensor(label > 0.5)

        return image, label

    def __len__(self):
        return len(self.img_dir)
    
    
class Train_Dataset1(Dataset):

    def __init__(self, img_list, label_list, transform=None, target_transform=None, image_resolution=None, tmf=get_aug_enhance()):

        self.img_dir = img_list
        self.label_dir = label_list

        self.transform = transform
        self.target_transform = target_transform
        self.image_resolution = image_resolution
        self.tmf = tmf

    def __getitem__(self, idx):

        image_name = self.img_dir[idx]
        label_name = self.label_dir[idx]

        img = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)

        img = np.array(img)
        label = np.array(label)

        augmented = self.tmf(image=img, mask=label)
        img, label = augmented['image'], augmented['mask']

        return img2tensor((img / 255.0 - mean) / std), img2tensor(label / 255.)


    def __len__(self):
        return len(self.img_dir)
    
    
class Valid_Dataset1(Dataset):

    def __init__(self, img_list, label_list, transform=None, target_transform=None, image_resolution=None):

        self.img_dir = img_list
        self.label_dir = label_list
        self.transform = transform
        self.target_transform = target_transform
        self.image_resolution = image_resolution

    def __getitem__(self, idx):

        image_name = self.img_dir[idx]
        label_name = self.label_dir[idx]

        img = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)

        img = np.array(img)
        label = np.array(label)

        return img2tensor((img / 255.0 - mean) / std), img2tensor(label / 255.)

    def __len__(self):
        return len(self.img_dir)




