import os
import time
import logging
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from scipy.ndimage import binary_closing, label
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from monai.metrics import HausdorffDistanceMetric
from thop import profile, clever_format
from model import model
from data import ACDC_transforms, AcdcDataset2D, ComposeTransforms

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.softmax(y_pred, dim=1)
        if y_true.dim() == 4 and y_true.shape[1] == 1:
            y_true = y_true.squeeze(1)

        y_true_one_hot = F.one_hot(y_true.long(), num_classes=y_pred.shape[1]).permute(0, 3, 1, 2).float()

        intersection = torch.sum(y_pred * y_true_one_hot, dim=(2, 3))
        sum_probs = torch.sum(y_pred, dim=(2, 3)) + torch.sum(y_true_one_hot, dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (sum_probs + self.smooth)
        return 1 - dice[:, 1:].mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha, beta, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.softmax(y_pred, dim=1)
        if y_true.dim() == 4 and y_true.shape[1] == 1:
            y_true = y_true.squeeze(1)

        y_true_one_hot = F.one_hot(y_true.long(), num_classes=y_pred.shape[1]).permute(0, 3, 1, 2).float()

        tp = torch.sum(y_pred * y_true_one_hot, dim=(2, 3))
        fp = torch.sum(y_pred * (1 - y_true_one_hot), dim=(2, 3))
        fn = torch.sum((1 - y_pred) * y_true_one_hot, dim=(2, 3))
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky_index[:, 1:].mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta)
        self.gamma = gamma

    def forward(self, logits, labels):
        return torch.pow(self.tversky(logits, labels), self.gamma)


class SuperCombinedLoss(nn.Module):
    def __init__(self, ce_w=0.5, dice_w=0.5, ft_w=0.0, alpha=0.7, beta=0.3, gamma=4.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.ft_loss = FocalTverskyLoss(alpha, beta, gamma)
        self.weights = (ce_w, dice_w, ft_w)

    def forward(self, logits, labels):
        labels_squeeze = labels.squeeze(1).long() if labels.dim() == 4 else labels.long()
        loss = self.weights[0] * self.ce_loss(logits, labels_squeeze)
        loss += self.weights[1] * self.dice_loss(logits, labels)
        if self.weights[2] > 0:
            loss += self.weights[2] * self.ft_loss(logits, labels)
        return loss

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, scaler, accumulation_steps=4):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    progress_bar = tqdm(dataloader, desc="Training")
    for i, batch in enumerate(progress_bar):
        images, labels = batch['image'].to(device), batch['label'].to(device)

        with autocast():
            outputs = model(images)

            if isinstance(outputs, (list, tuple)):
                loss_main = loss_fn(outputs[0], labels)
                labels_f = labels.float().unsqueeze(1) if labels.dim() == 3 else labels.float()

                loss_aux1 = loss_fn(outputs[1], F.max_pool2d(labels_f, 2, 2).long())
                loss_aux2 = loss_fn(outputs[2], F.max_pool2d(labels_f, 4, 4).long())
                loss_aux3 = loss_fn(outputs[3], F.max_pool2d(labels_f, 8, 8).long())

                total_loss = 1.0 * loss_main + 0.6 * loss_aux1 + 0.4 * loss_aux2 + 0.2 * loss_aux3
            else:
                total_loss = loss_fn(outputs, labels)

            loss_to_backward = total_loss / accumulation_steps
            scaler.scale(loss_to_backward).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            running_loss += total_loss.item()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            progress_bar.set_postfix(loss=f'{total_loss.item():.4f}')

    return running_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device, output_folder=None, epoch=0, save_freq=10):
    model.eval()

    class_dice_scores = {1: [], 2: [], 3: []}  # 1:RV, 2:MYO, 3:LV
    hd95_scores, iou_scores = [], []
    hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    for i, batch in enumerate(tqdm(dataloader, desc="Validating")):
        images, labels = batch['image'].to(device), batch['label'].to(device)
        outputs = model(images)

        labels_cpu = labels.cpu().squeeze(1) if labels.dim() == 4 else labels.cpu()

        if isinstance(outputs, (list, tuple)):
            probs_list = [torch.softmax(o.cpu(), dim=1) for o in outputs]
            fused_probs = torch.zeros_like(probs_list[0])
            weights = [0.8, 0.1, 0.06, 0.04]
            for j, probs in enumerate(probs_list):
                up_probs = F.interpolate(probs, size=labels_cpu.shape[-2:], mode='bilinear', align_corners=False)
                fused_probs += weights[j] * up_probs
            preds = torch.argmax(fused_probs, dim=1)
        else:
            preds = torch.argmax(torch.softmax(outputs.cpu(), dim=1), dim=1)

        for b_idx in range(preds.shape[0]):
            pred_np = preds[b_idx].numpy()
            label_np = labels_cpu[b_idx].numpy()
            num_classes = outputs[0].shape[1] if isinstance(outputs, (list, tuple)) else outputs.shape[1]

            pred_post = np.zeros_like(pred_np, dtype=np.uint8)
            MIN_PIXELS = {1: 10, 2: 10, 3: 15}

            for c in range(1, num_classes):
                class_mask = (pred_np == c)
                if not class_mask.any(): continue
                labeled_mask, num_features = label(class_mask)
                if num_features == 0: continue

                sizes = np.bincount(labeled_mask.ravel())[1:]
                if len(sizes) == 0: continue

                largest_idx = sizes.argmax() + 1
                if sizes.max() < MIN_PIXELS.get(c, 0): continue

                clean_mask = (labeled_mask == largest_idx)
                final_mask = binary_closing(clean_mask, structure=np.ones((3, 3)))
                pred_post[final_mask] = c

            p_oh = F.one_hot(torch.from_numpy(pred_post).long(), num_classes).permute(2, 0, 1).unsqueeze(0).float()
            l_oh = F.one_hot(torch.from_numpy(label_np).long(), num_classes).permute(2, 0, 1).unsqueeze(0).float()
            hd_vals = hd_metric(y_pred=p_oh, y=l_oh)

            for c_hd in hd_vals[0]:
                if not torch.isinf(c_hd) and not torch.isnan(c_hd):
                    hd95_scores.append(c_hd.item())
            for c in range(1, num_classes):
                p_c, l_c = (pred_post == c), (label_np == c)
                intersection = np.sum(p_c * l_c)
                union = np.sum(p_c) + np.sum(l_c) - intersection

                if union > 0:
                    # mIoU
                    iou = intersection / union
                    iou_scores.append(iou)

                    # Dice
                    dice = 2. * intersection / (np.sum(p_c) + np.sum(l_c) + 1e-8)
                    if c not in class_dice_scores:
                        class_dice_scores[c] = []
                    class_dice_scores[c].append(dice)

            if output_folder and b_idx == 0 and (i % save_freq == 0):
                save_2d_results(images[b_idx], label_np, pred_post, output_folder, epoch, i)

    mean_class_dice = {
        c: np.mean(scores) if scores else 0.0
        for c, scores in class_dice_scores.items()
    }
    overall_dice = np.mean(list(mean_class_dice.values())) if mean_class_dice else 0.0

    return {
        "dice": overall_dice,
        "dice_per_class": mean_class_dice, 
        "hd95": np.mean(hd95_scores) if hd95_scores else float('inf'),
        "miou": np.mean(iou_scores) if iou_scores else 0.0
    }


def save_2d_results(img, lbl, pred, folder, ep, b_idx):
    os.makedirs(folder, exist_ok=True)
    img_np = img.squeeze().cpu().numpy()
    if img_np.ndim == 3: img_np = img_np[0]

    sitk.WriteImage(sitk.GetImageFromArray(img_np), os.path.join(folder, f"ep{ep:03d}_{b_idx:03d}_img.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(lbl.astype(np.uint8)),
                    os.path.join(folder, f"ep{ep:03d}_{b_idx:03d}_gt.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(pred.astype(np.uint8)),
                    os.path.join(folder, f"ep{ep:03d}_{b_idx:03d}_pd.nii.gz"))


def setup_logger(path):
    logger = logging.getLogger('TrainLog')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S')
        fh, ch = logging.FileHandler(path), logging.StreamHandler()
        fh.setFormatter(fmt);
        ch.setFormatter(fmt)
        logger.addHandler(fh);
        logger.addHandler(ch)
    return logger


def main():
    DATASET = 'acdc'
    MODEL_TYPE = ''  #

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TRAIN_DATA_ROOT = '....'
    VAL_DATA_ROOT = '....'
    OUT_DIR = f"..."
    os.makedirs(OUT_DIR, exist_ok=True)

    logger = setup_logger(os.path.join(OUT_DIR, "train.log"))
    logger.info(f"Started {MODEL_TYPE} on {DEVICE}")
    BATCH_SIZE, NUM_WORKERS = 8, 4
    LR, EPOCHS, ACCUM = 6e-5, 200, 1
    TARGET_SIZE = (256, 256)
    if DATASET == 'acdc':
        train_transforms = ACDC_transforms(mode='train', target_size=TARGET_SIZE)
        val_transforms = ACDC_transforms(mode='test', target_size=TARGET_SIZE)

        train_dataset = AcdcDataset2D(data_root=TRAIN_DATA_ROOT, transform=train_transforms, mode='train')
        val_dataset = AcdcDataset2D(data_root=VAL_DATA_ROOT, transform=val_transforms, mode='val')

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
   
    model = model(in_channels=1, out_channels=4, base_c=32).to(DEVICE)
    dummy_input = torch.randn(1, 1, TARGET_SIZE[0], TARGET_SIZE[1]).to(DEVICE)
    try:
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        macs, params = clever_format([macs, params], "%.3f")
        logger.info(f"Model: {MODEL_TYPE} | Parameters: {params} | FLOPs (MACs): {macs}")
    except Exception as e:
        logger.warning(f"Failed to compute FLOPs: {e}")
        logger.info(f"Model: {MODEL_TYPE} | Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    loss_fn = SuperCombinedLoss(ce_w=0.5, dice_w=0.5).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler()
    best_dice = 0.0
    best_dice_per_class = {1: 0.0, 2: 0.0, 3: 0.0} 

    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{EPOCHS} | LR: {scheduler.get_last_lr()[0]:.2e}")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE, scaler, ACCUM)
        scheduler.step()

        val_metrics = validate(model, val_loader, DEVICE, os.path.join(OUT_DIR, "vis"), epoch + 1)

        logger.info(
            f"Train Loss: {train_loss:.4f} | Val mDice: {val_metrics['dice']:.4f} | "
            f"mIoU: {val_metrics['miou']:.4f} | HD95: {val_metrics['hd95']:.2f}"
        )

        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            best_dice_per_class = val_metrics['dice_per_class']
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))

            logger.info(f"Best Model Saved! (mDice: {best_dice:.4f} | "
                        f"RV: {best_dice_per_class.get(1, 0):.4f}, "
                        f"MYO: {best_dice_per_class.get(2, 0):.4f}, "
                        f"LV: {best_dice_per_class.get(3, 0):.4f})")

if __name__ == '__main__':
    main()
