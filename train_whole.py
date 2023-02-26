from __future__ import print_function, division
from dataset import Train_Dataset, Valid_Dataset
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
import warnings
import torch.nn.functional as F
from tqdm.contrib import tzip
from Net_BiT_Seg import Net, RGB
from Net_Swin_up import Net1
from Net_CSWin_Seg import Net2
from Net_Swin_Seg import Net3
from Net_HILA_BiT_Seg import Net4
from Net_BiT_DASeg import Net5
from Net_HRViT_Seg import Net6
from Net_pvt_v2 import Net7
from Net_CoaT import Net8
from lavaz_loss import lovasz_hinge2
import torch.cuda.amp as amp
is_amp = True  

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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
    use_scheduler = args.use_scheduler
    finetune = args.finetune
    finetune_path = args.finetune_path
    K = args.folds
    fold = args.k_th_fold
    use_carveMix = args.use_carveMix
    train_dataset_path = args.train_dataset_path
    train_gt_dataset_path = args.train_gt_dataset_path
    train_edge_dataset_path = args.train_edge_dataset_path
    valid_dataset_path = args.valid_dataset_path
    valid_gt_dataset_path = args.valid_gt_dataset_path
    New_folder = args.saved_model_path
    read_pred = args.visualize_of_data_aug_path
    weights_path = args.weights_path
    weights = args.weights
    weights1 = args.weights1
    
    # print params
    train_on_gpu = torch.cuda.is_available()
    print('*' * 20)
    print('Peng Lab')
    print('Network : ' + network)
    print(f'Fold {fold} / {K}')
    print(f'Training on GPU {device}')
    cuda = "cuda:" + str(device)
    device = torch.device(cuda if train_on_gpu else "cpu")
    print('image_size = ' + str(image_resolution))
    print('batch_size = ' + str(batch_size))
    print('epoch = ' + str(epoch))
    print('*' * 20)

    # initial params
    valid_loss_min = np.Inf
    best_metric = 0
    best_loss = 99999
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
    if network == "SegFormer":
        model_test = Net()
    if network == "upernet":
        model_test = Net1()
    if network == "CSWin":
        model_test = Net2()
    if network == "SwinSegFormer":
        model_test = Net3()
    if network == "HILASegFormer":
        model_test = Net4()
    if network == "DASegFormer":
        model_test = Net5()
    if network == "HRViT":
        model_test = Net6()
    if network == "PVT":
        model_test = Net7()
    if network == "CoaT":
        model_test = Net8()
    
    if network == "SegFormer" or network == "upernet" or network == "CSWin" or network == "SwinSegFormer" or network == "HILASegFormer" or network == "DASegFormer" or network == "PVT" or network == "CoaT":
        model_test.load_pretrain()
    if finetune:
        print("Start finetune!") 
        model_test.load_state_dict(torch.load(finetune_path))
    model_test.to(device)

    criterion = nn.BCEWithLogitsLoss()
    metric = Dice_soft()
    rgb = RGB()

    train_list = []
    train_list_GT = []
    train_list_edge = []
    for line in open(train_dataset_path).readlines():
        curLine = line.strip('\n')
        train_list.append(curLine)
    for line in open(train_gt_dataset_path).readlines():
        curLine = line.strip('\n')
        train_list_GT.append(curLine)
    for line in open(train_edge_dataset_path).readlines():
        curLine = line.strip('\n')
        train_list_edge.append(curLine)
        
    valid_list = []
    valid_list_GT = []
    for line in open(valid_dataset_path).readlines():
        curLine = line.strip('\n')
        valid_list.append(curLine)
    for line in open(valid_gt_dataset_path).readlines():
        curLine = line.strip('\n')
        valid_list_GT.append(curLine)

    print(f"{fold} / {K} fold training")

    # set DataLoader
    train_data = Train_Dataset(img_list=train_list, label_list=train_list_GT, edge_list=train_list_edge, image_resolution=image_resolution)
    valid_data = Valid_Dataset(img_list=valid_list, label_list=valid_list_GT, image_resolution=image_resolution)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=False)

    # set optimizer
    opt = torch.optim.AdamW(model_test.parameters(), lr=initial_lr, betas=(0.9, 0.999))
    if use_scheduler:
        t = 5  # warmup
        T = epoch
        n_t = 0.5
        lambda1 = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if n_t * (
                1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1 else n_t * (
                1 + math.cos(math.pi * (epoch - t) / (T - t)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)
    scaler = amp.GradScaler(enabled=is_amp)

    for i in range(epoch):
        train_loss = 0.0
        valid_loss = 0.0
        if use_scheduler:
            lr = scheduler.get_lr()
        model_test.train()
        if network == "SegFormer" or network == "upernet" or network == "CSWin" or network == "SwinSegFormer" or network == "HILASegFormer" or network == "DASegFormer" or network == "HRViT" or network == "PVT" or network == "CoaT":
            model_test.output_type = ['loss']
        k = 1
        for x, y, z, w in tqdm(train_loader):
            if use_carveMix:
                list = w.numpy()
                if list[0] in list[1:]:
                    selected_lidex = np.argwhere(list[1:] == list[0])
                    selected = selected_lidex[0] + 1
                    image1 = x[0].numpy()
                    label1 = y[0].numpy()
                    image2 = x[selected[0]].numpy()
                    label2 = y[selected[0]].numpy()
                    new_target, new_label = generate_new_sample(image1, image2, label1, label2)
                    x[0] = new_target
                    y[0] = new_label
            x, y, z, w = x.half().to(device), y.half().to(device), z.half().to(device), w.to(device)
            if network == "SegFormer" or network == "upernet" or network == "CSWin" or network == "SwinSegFormer" or network == "HILASegFormer" or network == "DASegFormer" or network == "HRViT" or network == "PVT" or network == "CoaT":
                input = {'image': x, 'mask': y, 'edge': z, 'cls': w}
            else:
                input = rgb(x)
            with amp.autocast(enabled=is_amp):
                output = model_test(input)
                if network == "SegFormer" or network == "upernet" or network == "CSWin" or network == "SwinSegFormer" or network == "HILASegFormer" or network == "DASegFormer" or network == "HRViT" or network == "PVT" or network == "CoaT":
                    loss0 = output['bce_loss'].mean()
                    loss1 = output['aux2_loss'].mean()
                    if finetune:
                        lossT = loss0 + 0.2 * loss1
                    else:
                        lossT = loss0 + 0.2 * loss1
                else:
                    lossT = calc_loss(output, y, bce_weight=0.7)
            opt.zero_grad()
            scaler.scale(lossT).backward()
            scaler.unscale_(opt)
            # torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
            scaler.step(opt)
            scaler.update()
            train_loss += lossT.item() * x.size(0)
            k = 2
        if use_scheduler:
            scheduler.step()
        train_loss = train_loss / len(train_list)
        print('Epoch: {}/{} Training Loss: {:.6f}'.format(i + 1, epoch, train_loss))

        valid_probability = []
        valid_mask = []
        model_test.eval()
        if network == "SegFormer" or network == "upernet" or network == "CSWin" or network == "SwinSegFormer" or network == "HILASegFormer" or network == "DASegFormer" or network == "HRViT" or network == "PVT" or network == "CoaT":
            model_test.output_type = ['inference']
        with torch.no_grad():
            with amp.autocast(enabled=is_amp):
                for x1, y1 in tqdm(valid_loader):
                    x1, y1 = x1.to(device), y1.to(device)
                    if network == "SegFormer" or network == "upernet" or network == "CSWin" or network == "SwinSegFormer" or network == "HILASegFormer" or network == "DASegFormer" or network == "HRViT" or network == "PVT" or network == "CoaT":
                        input = {'image': x1, 'mask': torch.ones((x1.shape[0], 1, x1.shape[2], x1.shape[3])).to(device)}
                    else:
                        input = rgb(x1)
                    output = model_test(input)
                    if network == "SegFormer" or network == "upernet" or network == "CSWin" or network == "SwinSegFormer" or network == "HILASegFormer" or network == "DASegFormer" or network == "HRViT" or network == "PVT" or network == "CoaT":
                        output['probability'] = F.interpolate(output['probability'], size=(3000, 3000), mode='bilinear', align_corners=False)
                        pred = torch.where(output['probability'] > 0.5, 1, 0)
                        # valid_loss = output['valid_loss'].data.cpu().numpy()
                    else:
                        output = torch.sigmoid(output)
                        # output = F.interpolate(output, size=(3000, 3000), mode='bilinear', align_corners=False)
                        pred = torch.where(output > 0.5, 1, 0)
                    valid_probability.append(pred.data.cpu().numpy())
                    valid_mask.append(y1.data.cpu().numpy())
        probability = np.concatenate(valid_probability)
        mask = np.concatenate(valid_mask)
        dice = compute_dice_score(probability, mask)
        metric_this_epoch = dice.mean()
        
        if metric_this_epoch > best_metric:
            print('Validation dice increased ({:.6f} --> {:.6f}). Saving model! '.format(best_metric, metric_this_epoch))
            torch.save(model_test.state_dict(), weights)
            best_metric = metric_this_epoch
        else:
            print('Validation dice decreased ({:.6f} <-- {:.6f}). '.format(best_metric, metric_this_epoch))
            
#         if valid_loss < best_loss:
#             print('Validation loss decreased ({:.6f} --> {:.6f}). Saving model! '.format(best_loss, valid_loss))
#             torch.save(model_test.state_dict(), weights1)
#             best_loss = valid_loss
#         else:
#             print('Validation loss increased ({:.6f} <-- {:.6f}). '.format(best_loss, valid_loss))
            
        torch.save(model_test.state_dict(), f'/root/autodl-tmp/best_model/latest_model_{network}_{fold}_lung.pth')


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description="HuBMAP", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument('--input_channel', type=int, default=3, help='image channel')
    parser.add_argument('--output_class', type=int, default=1, help='output class, binary classification (output_class = 1)')
    parser.add_argument('--image_resolution', type=int, default=768, help='image resolution we resize')
    
    parser.add_argument('--epochs', type=int, default=200, help='max epoch')
    parser.add_argument('--batch_size', type=int, default=5, help='batch size')
    parser.add_argument('--backbone', type=str, default="efficientnet-b7", help='backbone')
    parser.add_argument('--network', type=str, default="Unet", help='network')
    parser.add_argument('--initial_learning_rate', type=float, default=7.5e-4, help='initial learning rate')
    parser.add_argument('--use_scheduler', type=bool, default=True, help='use scheduler')
    parser.add_argument('--use_carveMix', type=bool, default=False, help='use carveMix')
    
    parser.add_argument('--finetune', type=bool, default=False, help='finetune model')
    parser.add_argument('--finetune_path', type=str, default=r'/root/HuBMAP/code/saved_model/weights/best_model_segbit2v2_1_7949.pth', help='finetune model path')
    
    parser.add_argument('--folds', type=int, default=5, help='split number')
    parser.add_argument('--k_th_fold', type=int, default=1, help='k-th fold we train')
    
    # kidney organ
#     parser.add_argument('--train_dataset_path', type=str, default=r'/root/HuBMAP/code/train_valid_list_path_newprostate/train_img_fold_1.txt', help='train dataset path')
#     parser.add_argument('--train_gt_dataset_path', type=str, default=r'/root/HuBMAP/code/train_valid_list_path_newprostate/train_mask_fold_1.txt', help='train ground truth path')
#     parser.add_argument('--train_edge_dataset_path', type=str, default=r'/root/autodl-tmp/train_edge_768_x3_fold_1_160all.txt', help='train edge path')
#     parser.add_argument('--valid_dataset_path', type=str, default=r'/root/HuBMAP/code/train_valid_list_path_newprostate/valid_img_fold_1.txt', help='valid dataset path')
#     parser.add_argument('--valid_gt_dataset_path', type=str, default=r'/root/HuBMAP/code/train_valid_list_path_newprostate/valid_mask_fold_1.txt', help='valid ground truth path')
    
    # no lung organ
    parser.add_argument('--train_dataset_path', type=str, default=r'/root/autodl-tmp/train_img_768_x3_fold_1_lung.txt', help='train dataset path')
    parser.add_argument('--train_gt_dataset_path', type=str, default=r'/root/autodl-tmp/train_mask_768_x3_fold_1_lung.txt', help='train ground truth path')
    parser.add_argument('--train_edge_dataset_path', type=str, default=r'/root/autodl-tmp/train_edge_768_x3_fold_2_prostate.txt', help='train edge path')
    parser.add_argument('--valid_dataset_path', type=str, default=r'/root/autodl-tmp/valid_img_768_x3_fold_1_lung.txt', help='valid dataset path')
    parser.add_argument('--valid_gt_dataset_path', type=str, default=r'/root/autodl-tmp/valid_mask_768_x3_fold_1_lung.txt', help='valid ground truth path')
    
    parser.add_argument('--saved_model_path', type=str, default="./saved_model", help='saved model path')
    parser.add_argument('--visualize_of_data_aug_path', type=str, default="./saved_model/pred", help='visualization data augmentation')
    parser.add_argument('--weights_path', type=str, default="./saved_model/weights", help='weights path')
    parser.add_argument('--weights', type=str, default=r'/root/autodl-tmp/best_model/best_model_eff7u_1_lung_dice.pth', help='best_model.pth')
    parser.add_argument('--weights1', type=str, default=r'/root/autodl-tmp/best_model/best_model_eff7u_1_lung_dice.pth', help='best_model.pth')

    args, unkown = parser.parse_known_args()
    main(args)
    
    
    
    
    
    
    
    
    
    