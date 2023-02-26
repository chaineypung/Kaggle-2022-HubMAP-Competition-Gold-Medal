from __future__ import print_function, division
from dataset import Train_Dataset1, Valid_Dataset1
from torch.utils.data import DataLoader
import shutil
import argparse
from losses import calc_loss, dice_loss, threshold_predictions_v, threshold_predictions_p, Dice_soft, cutout, compute_dice_score
from ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow
from metrics import *
from util import *
import segmentation_models_pytorch as smp
import math
import rasterio
from rasterio.windows import Window
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
import torch.nn.functional as F
import warnings
from tqdm.contrib import tzip
from PIL import Image

warnings.filterwarnings("ignore")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

sz = 256
sz_reduction = 2
expansion = 256
TH = 0.5
bs = 4
s_th = 40
p_th = 1000 * (sz // 256) ** 2
identity = rasterio.Affine(1, 0, 0, 0, 1, 0)

# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])

mean = np.array([0.7720342, 0.74582646, 0.76392896])
std = np.array([0.24745085, 0.26182273, 0.25782376])

# mean = np.array([0.79117177, 0.7583153, 0.78292631])
# std = np.array([[0.16705585, 0.19441158, 0.18427182]])

img_txt = r'./train_valid_list_path/valid_img_fold_2.txt'
mask_txt = r'./train_valid_list_path/valid_mask_fold_2.txt'

f = open(img_txt, 'r')
img_list = list(f)
f.close()
f = open(mask_txt, 'r')
mask_list = list(f)
f.close()

def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2: img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


class HuBMAPDataset(Dataset):

    def __init__(self, img_idx, sz=sz, sz_reduction=sz_reduction, expansion=expansion):
        self.data = rasterio.open(img_idx, transform=identity, num_threads='all_cpus')
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

        n0, n1 = idx // self.n1max, idx % self.n1max
        x0, y0 = -self.pad0 // 2 + n0 * self.sz - self.expansion // 2, -self.pad1 // 2 + n1 * self.sz - self.expansion // 2
        p00, p01 = max(0, x0), min(x0 + self.sz + self.expansion, self.shape[0])
        p10, p11 = max(0, y0), min(y0 + self.sz + self.expansion, self.shape[1])
        img = np.zeros((self.sz + self.expansion, self.sz + self.expansion, 3), np.uint8)
        if self.data.count == 3:
            img[(p00 - x0):(p01 - x0), (p10 - y0):(p11 - y0)] = np.moveaxis(self.data.read([1, 2, 3],
                                                                                           window=Window.from_slices(
                                                                                               (p00, p01), (p10, p11))),
                                                                            0, -1)
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


class Model_pred:
    def __init__(self, models, dl, sz_reduction=sz_reduction, tta: bool = True, half: bool = False, device=None):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half
        self.device = device
        self.sz_reduction = sz_reduction

    def __iter__(self):
        count = 0
        with torch.no_grad():
            for x, y in iter(self.dl):
                if ((y >= 0).sum() > 0):
                    x = x[y >= 0].to(self.device)
                    y = y[y >= 0]
                    if self.half: x = x.half()
                    py = None
                    for model in self.models:
                        p = model(x)
                        p = torch.sigmoid(p).detach()
                        if py is None:
                            py = p
                        else:
                            py += p
                    if self.tta:
                        flips = [[-1], [-2], [-2, -1]]
                        for f in flips:
                            xf = torch.flip(x, f)
                            for model in self.models:
                                p = model(xf)
                                p = torch.flip(p, f)
                                py += torch.sigmoid(p).detach()
                        py /= (1 + len(flips))
                    py /= len(self.models)
                    py = F.upsample(py, scale_factor=self.sz_reduction, mode="bilinear")
                    py = py.permute(0, 2, 3, 1).float().cpu()
                    batch_size = len(py)
                    for i in range(batch_size):
                        yield py[i, expansion * self.sz_reduction // 2:-expansion * self.sz_reduction // 2,
                              expansion * self.sz_reduction // 2:-expansion * self.sz_reduction // 2], y[i]
                        count += 1

    def __len__(self):
        return len(self.dl.dataset)


def main(args):
    # load args
    input_channel = args.input_channel
    output_class = args.output_class
    image_resolution = args.image_resolution
    epoch = args.epochs
    num_workers = args.num_workers
    device = args.device
    batch_size = args.batch_size
    backbone = args.backbone
    network = args.network
    initial_lr = args.initial_learning_rate
    MAX_STEP = args.t_max
    K = args.folds
    fold = args.k_th_fold
    fold_file_list = args.fold_file_list
    train_dataset_path = args.train_dataset_path
    train_gt_dataset_path = args.train_gt_dataset_path
    New_folder = args.saved_model_path
    read_pred = args.visualize_of_data_aug_path
    weights_path = args.weights_path
    weights = args.weights

    # check GPU
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('Training on CPU')
    else:
        print(f'Training on GPU {device}')
    cuda = "cuda:" + str(device)
    device = torch.device(cuda if train_on_gpu else "cpu")
    print('image_size = ' + str(image_resolution))
    print('batch_size = ' + str(batch_size))
    print('epoch = ' + str(epoch))

    # initial params
    valid_loss_min = np.Inf
    best_metric = 0
    lossT, lossL = [], []
    lossL.append(np.inf)
    lossT.append(np.inf)
    epoch_valid = epoch - 2
    n_iter, i_valid, model_test = 1, 0, 0

    # set pin_memory
    pin_memory = False
    if train_on_gpu:
        pin_memory = True

    # select backbone and network
    if network == "Linknet":
        model_test = smp.Linknet(encoder_name=backbone, encoder_weights='imagenet', in_channels=input_channel,
                                 classes=output_class)
    if network == "DeepLabV3Plus":
        model_test = smp.DeepLabV3Plus(encoder_name=backbone, encoder_weights='imagenet', in_channels=input_channel,
                                       classes=output_class)
    if network == "FPN":
        model_test = smp.FPN(encoder_name=backbone, encoder_weights='imagenet', in_channels=input_channel,
                             classes=output_class)
    if network == "PAN":
        model_test = smp.PAN(encoder_name=backbone, encoder_weights='imagenet', in_channels=input_channel,
                             classes=output_class)
    if network == "PSPNet":
        model_test = smp.PSPNet(encoder_name=backbone, encoder_weights='imagenet', in_channels=input_channel,
                                classes=output_class)
    if network == "Unet":
        model_test = smp.Unet(encoder_name=backbone, encoder_weights='imagenet', in_channels=input_channel,
                              classes=output_class)
    if network == "unext101":
        model_test = UneXt50()

    #     print("Load pretrained model!") model_test.load_state_dict(torch.load("./saved_model/weights/best_model_0.pth"), map_location='cuda:0')
    model_test.to(device)

    criterion = nn.BCEWithLogitsLoss()
    metric = Dice_soft()

    train_list = []
    train_list_GT = []
    for line in open(train_dataset_path).readlines():
        curLine = line.strip('\n')
        train_list.append(curLine)
    for line in open(train_gt_dataset_path).readlines():
        curLine = line.strip('\n')
        train_list_GT.append(curLine)

    print(f"{fold} / {K} fold training")

    # set DataLoader
    train_data = Train_Dataset1(img_list=train_list, label_list=train_list_GT, image_resolution=image_resolution)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=False)

    # set optimizer
    opt = torch.optim.AdamW(model_test.parameters(), lr=initial_lr, betas=(0.9, 0.999), weight_decay=1e-2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(MAX_STEP), eta_min=1e-11)

    t = 5  # warmup
    T = epoch
    n_t = 0.5
    lambda1 = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if n_t * (
            1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1 else n_t * (
            1 + math.cos(math.pi * (epoch - t) / (T - t)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)

    for i in range(epoch):

        train_loss = 0.0
        valid_loss = 0.0
        lr = scheduler.get_lr()
        model_test.train()
        k = 1
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            # x = cutout(x)
            input_images(x, y, i, n_iter, k)
            y_pred = model_test(x)
            lossT = calc_loss(y_pred, y, bce_weight=0.5)
            opt.zero_grad()
            lossT.backward()
            opt.step()
            train_loss += lossT.item() * x.size(0)
            k = 2
        scheduler.step()
        train_loss = train_loss / len(train_list)
        print('Epoch: {}/{} Training Loss: {:.6f} Learning Rate: {:.9f}'.format(i + 1, epoch, train_loss, lr[0]))
        
        valid_probability = []
        valid_mask = []
        model_test.eval()
        for img_index, mask_index in tzip(img_list, mask_list):
            img_index = img_index.strip('\n')
            mask_index = mask_index.strip('\n')
            index = int(img_index.split("images/")[1].split(".tiff")[0])
            ds = HuBMAPDataset(img_idx=img_index)
            # rasterio cannot be used with multiple workers
            dl = DataLoader(ds, bs, num_workers=0, shuffle=False, pin_memory=True)
            mp = Model_pred([model_test], dl, device=device)
            # generate masks
            mask = torch.zeros(len(ds), ds.sz, ds.sz, dtype=torch.int8).to(device)
            for p, ii in iter(mp): mask[ii.item()] = p.squeeze(-1) > TH
            # reshape tiled masks into a single mask and crop padding
            mask = mask.view(ds.n0max, ds.n1max, ds.sz, ds.sz). \
                permute(0, 2, 1, 3).reshape(ds.n0max * ds.sz, ds.n1max * ds.sz)
            mask = mask[ds.pad0 // 2:-(ds.pad0 - ds.pad0 // 2) if ds.pad0 > 0 else ds.n0max * ds.sz,
                   ds.pad1 // 2:-(ds.pad1 - ds.pad1 // 2) if ds.pad1 > 0 else ds.n1max * ds.sz]
            gt = torch.from_numpy(np.asarray(Image.open(mask_index))) / 255.
            gt = gt.to(device)
            if gt.shape[0] != 3000:
                continue
            mask = torch.where(mask > 0.5, 1, 0)
            valid_probability.append(mask.data.cpu().numpy())
            valid_mask.append(gt.data.cpu().numpy())
        probability = np.concatenate(valid_probability)
        mask = np.concatenate(valid_mask)
        dice = compute_dice_score(probability, mask)
        metric_this_epoch = dice.mean()
        if metric_this_epoch > best_metric:
            print('Validation dice increased ({:.6f} --> {:.6f}). Saving model! '.format(best_metric, metric_this_epoch))
            torch.save(model_test.state_dict(), weights)
            best_metric = metric_this_epoch
        else:
            print('Validation dice decrease ({:.6f} <-- {:.6f}). '.format(best_metric, metric_this_epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="HuBMAP", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_channel', type=int, default=3, help='image channel')
    parser.add_argument('--output_class', type=int, default=1,
                        help='output class, binary classification (output_class = 1)')
    parser.add_argument('--image_resolution', type=int, default=256, help='image resolution we resize')
    parser.add_argument('--epochs', type=int, default=60, help='max epoch')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--backbone', type=str, default="efficientnet-b3", help='backbone')
    parser.add_argument('--network', type=str, default="Linknet", help='network')
    parser.add_argument('--initial_learning_rate', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--t_max', type=int, default=40, help='CosineAnnealingLR parameter')
    parser.add_argument('--folds', type=int, default=5, help='split number')
    parser.add_argument('--k_th_fold', type=int, default=2, help='k-th fold we train')

    parser.add_argument('--fold_file_list', type=str, default=r'/root/autodl-tmp/train_cut_256.csv',
                        help='fold file list')
    parser.add_argument('--train_dataset_path', type=str, default=r'/root/autodl-tmp/train_img_256_x2_fold_2.txt',
                        help='train dataset path')
    parser.add_argument('--train_gt_dataset_path', type=str, default=r'/root/autodl-tmp/train_mask_256_x2_fold_2.txt',
                        help='train ground truth path')
    parser.add_argument('--saved_model_path', type=str, default="./saved_model", help='saved model path')
    parser.add_argument('--visualize_of_data_aug_path', type=str, default="./saved_model/pred",
                        help='visualization data augmentation')
    parser.add_argument('--weights_path', type=str, default="./saved_model/weights", help='weights path')
    parser.add_argument('--weights', type=str, default="./saved_model/weights/best_model_linkb3_256_2.pth",
                        help='best_model.pth')

    args, unkown = parser.parse_known_args()
    main(args)


