import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import math
import logging
import time as _time
import numpy as np
from ACDC.file_2D.model_1 import HDC_Net
# 导入 thop 用于计算参数量和 FLOPs
from thop import profile, clever_format
from monai.networks.nets import UNet, AttentionUnet
from Compare.models import TransUNet, SwinUNet
from Compare.FocalTransNet import Net as FocalTransNet
from data_31 import (SynapseNpz2dDataset, SynapseH5ValDataset, ZScoreNormalization2D, Resize2D,
                     RandomFlip2D, RandomRotate2D, ToTensor2D, ComposeTransforms, MinMaxToMinusOneOne,
                     RandomGaussianNoise2D, RandomScale2D)
from monai.metrics import HausdorffDistanceMetric
from torch.cuda.amp import GradScaler, autocast
import SimpleITK as sitk
from monai.losses import DiceCELoss

# ==========================================
# 1. Loss & Utils
# ==========================================
# ==========================================
# 1. 修复版的 Batch-level DiceLoss (保命神器)
# ==========================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.softmax(y_pred, dim=1)
        # One-hot encoding
        y_true_one_hot = nn.functional.one_hot(y_true.long(), num_classes=y_pred.shape[1]).permute(0, 3, 1, 2).float()
        intersection = torch.sum(y_pred * y_true_one_hot, dim=(0, 2, 3))
        sum_probs = torch.sum(y_pred, dim=(0, 2, 3)) + torch.sum(y_true_one_hot, dim=(0, 2, 3))

        dice = (2. * intersection + self.smooth) / (sum_probs + self.smooth)
        return 1 - dice[1:].mean()


class SuperCombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super(SuperCombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, y_pred_logits, y_true_labels):
        if y_true_labels.dim() == 4 and y_true_labels.shape[1] == 1:
            y_true_labels = y_true_labels.squeeze(1)
        loss_ce = self.ce_loss(y_pred_logits, y_true_labels.long())
        loss_dice = self.dice_loss(y_pred_logits, y_true_labels)
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice


def setup_logger(log_file_path):
    logger = logging.getLogger('TrainingLog')
    if logger.hasHandlers(): logger.handlers.clear()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file_path, mode='a')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


SYNAPSE_CLASSES = {1: 'Aorta', 2: 'Gallbladder', 3: 'Kidney(L)', 4: 'Kidney(R)', 5: 'Liver', 6: 'Pancreas', 7: 'Spleen',
                   8: 'Stomach'}


# ==========================================
# 2. 训练与验证引擎
# ==========================================
def train_one_epoch_2d(model, dataloader, optimizer, loss_fn, device, scaler):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training 2D")

    for batch in progress_bar:
        images, labels = batch['image'].to(device), batch['label'].to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            if labels.dim() == 3:
                labels = labels.unsqueeze(1)  # 变成 [B, 1, H, W]

            # 【核心修改】：开启深监督 (Deep Supervision)
            if isinstance(outputs, (list, tuple)) and len(outputs) > 1:
                loss = 0.0
                # 赋予不同尺度的输出不同的权重 (主输出权重最高)
                weights = [1.0, 0.6, 0.4, 0.2]
                for idx, out in enumerate(outputs):
                    weight = weights[idx] if idx < len(weights) else 0.1
                    # 如果辅助输出尺寸变小了，将 label 缩小至对应尺寸 (必须用 nearest 保证类别不变)
                    if out.shape[-2:] != labels.shape[-2:]:
                        target_label = torch.nn.functional.interpolate(
                            labels.float(), size=out.shape[-2:], mode='nearest'
                        ).long()
                    else:
                        target_label = labels

                    loss += weight * loss_fn(out, target_label)
            else:
                # 单输出模型兼容
                if isinstance(outputs, (list, tuple)): outputs = outputs[0]
                loss = loss_fn(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')

    return running_loss / len(dataloader)


def validate_2d(model, dataloader, device, num_classes, output_folder=None):
    model.eval()
    class_dice_scores = {c: [] for c in range(1, num_classes)}
    class_iou_scores = {c: [] for c in range(1, num_classes)}
    hd95_scores = []
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    if output_folder: os.makedirs(output_folder, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validating")):
            image_vol, label_vol, case_name = batch['image'], batch['label'], batch['case_name'][0]
            image_slices = image_vol.squeeze(0)

            slice_preds = []
            for i in range(0, image_slices.shape[0], 16):
                slice_batch = image_slices[i:i + 16].to(device)
                outputs = model(slice_batch)
                outputs = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
                preds = torch.argmax(outputs, dim=1)
                slice_preds.append(preds.cpu())

            pred_vol = torch.cat(slice_preds, dim=0)
            label_vol = label_vol.squeeze(0)

            for c in range(1, num_classes):
                p_c, l_c = (pred_vol == c), (label_vol == c)
                intersection = (p_c & l_c).sum().float()
                union = (p_c | l_c).sum().float()
                if union > 0:
                    class_iou_scores[c].append((intersection / union).item())
                    class_dice_scores[c].append((2.0 * intersection / (p_c.sum() + l_c.sum() + 1e-8)).item())

            pred_one_hot = nn.functional.one_hot(pred_vol, num_classes=num_classes).permute(3, 0, 1, 2).unsqueeze(
                0).float()
            label_one_hot = nn.functional.one_hot(label_vol, num_classes=num_classes).permute(3, 0, 1, 2).unsqueeze(
                0).float()
            hd95_vals = hd95_metric(y_pred=pred_one_hot, y=label_one_hot)
            for hd_val in hd95_vals[0]:
                if not torch.isinf(hd_val) and not torch.isnan(hd_val):
                    hd95_scores.append(hd_val.item())

            if output_folder and batch_idx < 4:
                img_np = image_vol.squeeze(0).squeeze(1).cpu().numpy()
                pred_np = pred_vol.cpu().numpy().astype(np.uint8)
                label_np = label_vol.cpu().numpy().astype(np.uint8)

                fg_counts = (label_np > 0).sum(axis=(1, 2))
                best_z_idx = fg_counts.argmax()

                if fg_counts[best_z_idx] > 0:
                    sitk.WriteImage(sitk.GetImageFromArray(img_np[best_z_idx]),
                                    os.path.join(output_folder, f"{case_name}_slice{best_z_idx:03d}_img.nii.gz"))
                    sitk.WriteImage(sitk.GetImageFromArray(pred_np[best_z_idx]),
                                    os.path.join(output_folder, f"{case_name}_slice{best_z_idx:03d}_pred.nii.gz"))
                    sitk.WriteImage(sitk.GetImageFromArray(label_np[best_z_idx]),
                                    os.path.join(output_folder, f"{case_name}_slice{best_z_idx:03d}_gt.nii.gz"))

    mean_class_dice = {c: np.mean(scores) if scores else 0.0 for c, scores in class_dice_scores.items()}
    mean_class_iou = {c: np.mean(scores) if scores else 0.0 for c, scores in class_iou_scores.items()}
    return {
        "dice": np.mean(list(mean_class_dice.values())),
        "miou": np.mean(list(mean_class_iou.values())),
        "hd95": np.mean(hd95_scores) if hd95_scores else float('inf'),
        "dice_per_class": mean_class_dice
    }


# ==========================================
# 3. Main Script
# ==========================================
def main():
    # torch.backends.cudnn.benchmark = True

    SYNAPSE_DATA_ROOT = "/home/zgm/ZHF/HDC/Synapse/data"
    BASE_OUTPUT_DIR = "./runs_synapse_npz_2d"
    # MODEL_NAME = "FocalTransNet"
    MODEL_NAME = "FGUNet"
    RUN_ID = f"synapse2d_{MODEL_NAME}_{_time.strftime('%Y%m%d-%H%M')}"
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, RUN_ID)
    VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)

    logger = setup_logger(os.path.join(OUTPUT_DIR, "training_log.txt"))

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 8
    NUM_WORKERS = 6
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 200
    WARMUP_EPOCHS = 15  # 新增：预热轮数

    NUM_CLASSES = 9
    IMG_SIZE = (224, 224)

    logger.info("📦 Preparing Synapse 2D NPZ dataset...")

    train_transforms = ComposeTransforms([
        Resize2D(IMG_SIZE), RandomFlip2D(prob=0.5), RandomRotate2D(angle_range=(-10, 10)),
        RandomScale2D(prob=0.5, scale_range=(0.8, 1.2)),
        RandomGaussianNoise2D(prob=0.3, std=0.02),
        MinMaxToMinusOneOne(), ToTensor2D()
    ])
    val_transforms = ComposeTransforms([Resize2D(IMG_SIZE), MinMaxToMinusOneOne(), ToTensor2D()])

    train_dataset = SynapseNpz2dDataset(base_dir=os.path.join(SYNAPSE_DATA_ROOT, 'train_npz'),
                                        list_dir=SYNAPSE_DATA_ROOT, transform=train_transforms)
    val_dataset = SynapseH5ValDataset(base_dir=os.path.join(SYNAPSE_DATA_ROOT, 'test_vol_h5'),
                                      list_dir=SYNAPSE_DATA_ROOT, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    # model = FocalTransNet(img_size=IMG_SIZE[0], dim_in=1, dim_out=NUM_CLASSES, device=str(DEVICE)).to(DEVICE)
    # model = HDC_Net(in_channels=1, out_channels=9, base_c=32).to(DEVICE)
    # model = UNet(spatial_dims=2, in_channels=1, out_channels=9,channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=2).to(DEVICE)
    # model = MERIT_Cascaded(n_class=9, img_size_s2=(224, 224))
    # model = FG(seg_classes=9).to(DEVICE)
    dummy_input = torch.randn(1, 1, IMG_SIZE[0], IMG_SIZE[1]).to(DEVICE)
    try:
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        macs, params = clever_format([macs, params], "%.3f")
        logger.info(f"Model: {MODEL_NAME} | Parameters: {params} | FLOPs (MACs): {macs}")
    except Exception as e:
        logger.info(f"Model: {MODEL_NAME} | Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False,
                         squared_pred=True, batch=True, lambda_ce=0.5, lambda_dice=0.5).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            # 前 15 轮，学习率从 0 线性增长到 1.0 * LEARNING_RATE
            return float(epoch + 1) / float(max(1, WARMUP_EPOCHS))
        # 之后使用余弦退火衰减
        progress = float(epoch - WARMUP_EPOCHS) / float(max(1, NUM_EPOCHS - WARMUP_EPOCHS))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    scaler = GradScaler()

    best_val_dice = 0.0
    best_dice_per_class = {}

    logger.info("--- Starting 2D Training ---")
    for epoch in range(NUM_EPOCHS):
        logger.info(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} | LR: {scheduler.get_last_lr()[0] * LEARNING_RATE:.2e} ---")

        train_loss = train_one_epoch_2d(model, train_loader, optimizer, loss_fn, DEVICE, scaler)
        scheduler.step()

        logger.info(f"Train Loss: {train_loss:.4f}")

        if (epoch + 1) % 2 == 0 or epoch == NUM_EPOCHS - 1:
            epoch_viz_folder = os.path.join(VISUALIZATION_DIR, f"epoch_{epoch + 1}")
            val_metrics = validate_2d(model, val_loader, DEVICE, num_classes=NUM_CLASSES,
                                      output_folder=epoch_viz_folder)

            logger.info(
                f"Validation 3D Metrics -> mDice: {val_metrics['dice']:.4f} | mIoU: {val_metrics['miou']:.4f} | HD95: {val_metrics['hd95']:.4f}")

            if val_metrics['dice'] > best_val_dice:
                best_val_dice = val_metrics['dice']
                best_dice_per_class = val_metrics['dice_per_class']
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_2d_model.pth"))
                logger.info(f"🌟 Saved new best model with mDice: {best_val_dice:.4f}")

    logger.info("\n" + "=" * 50)
    logger.info(f"Training Finished! Model: {MODEL_NAME}")
    logger.info(f"Best Validation 3D mDice: {best_val_dice:.4f}")
    logger.info("--- 详细类别 Dice 分数 ---")
    for class_idx, class_name in SYNAPSE_CLASSES.items():
        logger.info(f"   - {class_name:<12} (Class {class_idx}): {best_dice_per_class.get(class_idx, 0.0):.4f}")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()