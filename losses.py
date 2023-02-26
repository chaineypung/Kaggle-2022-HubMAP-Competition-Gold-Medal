from __future__ import print_function, division
import torch.nn.functional as F
import torch
from fastai.vision.all import *

class Focal_loss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True):
        super(Focal_loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, logits, label):
        # logits:[b,h,w] label:[b,h,w]
        logits = torch.squeeze(logits)
        label = torch.squeeze(label)
        pred = logits.sigmoid()
        pred = pred.view(-1)  # b*h*w
        label = label.view(-1)

        if self.alpha:
#             self.alpha = self.alpha.type_as(pred.data)
            alpha_t = self.alpha * label + (1 - self.alpha) * (1 - label)  # b*h*w

        pt = pred * label + (1 - pred) * (1 - label)
        diff = (1 - pt) ** self.gamma

        FL = -1 * alpha_t * diff * pt.log()

        if self.size_average:
            return FL.mean()
        else:
            return FL.sum()
        
def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))

def compute_dice_score(probability, mask):
    N = len(probability)
    p = probability.reshape(N,-1)
    t = mask.reshape(N,-1)

    p = p > 0.5
    t = t > 0.5
    uion = p.sum(-1) + t.sum(-1)
    overlap = (p*t).sum(-1)
    dice = 2*overlap/(uion+0.0001)
    return dice

def cutout(tensor,alpha=0.5):
    x=int(alpha*tensor.shape[2])
    y=int(alpha*tensor.shape[3])
    center=np.random.randint(0,tensor.shape[2],size=(2))
    #perm = torch.randperm(img.shape[0])
    cut_tensor=tensor.clone()
    cut_tensor[:,:,center[0]-x//4:center[0]+x//4,center[1]-y//4:center[1]+y//4]=0
    return cut_tensor

class Dice_soft(Metric):
    def __init__(self, axis=1):
        self.axis = axis
    def reset(self): self.dice = 0
    def accumulate(self, pred, targ):
#         pred = torch.sigmoid(pred)
#         self.inter += (pred*targ).float().sum().item()
#         self.union += (pred+targ).float().sum().item()
        N = len(pred)
        p = pred.reshape(N,-1)
        t = targ.reshape(N,-1)
        p = p>0.5
        t = t>0.5
        uion = p.sum(-1) + t.sum(-1)
        overlap = (p*t).sum(-1)
        self.dice = 2*overlap/(uion+0.0001)
    @property
    def value(self): return self.dice if self.dice > 0 else None

def calc_loss(prediction, target, bce_weight=0.0):# 0.2
    """Calculating the loss and metrics
    Args:
        prediction = predicted images
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = F.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss

def threshold_predictions_v(predictions, thr=150):
    thresholded_preds = predictions[:]
   # hist = cv2.calcHist([predictions], [0], None, [2], [0, 2])
   # plt.plot(hist)
   # plt.xlim([0, 2])
   # plt.show()
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 255
    return thresholded_preds

def threshold_predictions_p(predictions, thr=0.01):
    thresholded_preds = predictions[:]
    #hist = cv2.calcHist([predictions], [0], None, [256], [0, 256])
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds