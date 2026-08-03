import glob
import os
import time
import logging
import cv2
from tqdm import tqdm
from scipy.ndimage import binary_closing, label
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import math
import random
import numpy as np
import torch
import imgaug.augmenters as iaa
import torch.nn.functional as F
from torch.utils.data import Dataset
from monai.metrics import HausdorffDistanceMetric
from monai.networks.nets import UNet, AttentionUnet
from thop import profile, clever_format
from model import model


def mask_to_onehot(mask):
    semantic_map = []
    mask = np.expand_dims(mask,-1)
    for colour in range (9):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.int32)
    return semantic_map
def augment_seg(img_aug, img, seg ):
    seg = mask_to_onehot(seg)
    aug_det = img_aug.to_deterministic()
    image_aug = aug_det.augment_image( img )

    segmap = ia.SegmentationMapsOnImage( seg, shape=img.shape )
    segmap_aug = aug_det.augment_segmentation_maps( segmap )
    segmap_aug = segmap_aug.get_arr()
    segmap_aug = np.argmax(segmap_aug, axis=-1).astype(np.float32)
    return image_aug , segmap_aug


class ISIC2018_dataset(Dataset):
    def __init__(self, base_dir, split, img_size, norm_x_transform=None, norm_y_transform=None, shuffle=True,
                 test_mode=False) -> None:
        super().__init__()
        self.norm_x_transform = norm_x_transform
        self.norm_y_transform = norm_y_transform
        self.data_dir = os.path.join(base_dir, split)
        self.split = split
        self.sample_list = os.listdir(self.data_dir)

        if test_mode:
            self.sample_list = self.sample_list[:int(len(self.sample_list) * 0.1)]

        if shuffle:
            random.shuffle(self.sample_list)

        self.img_size = img_size

        self.img_aug = iaa.SomeOf((0, 4), [
            iaa.Flipud(0.5, name="Flipud"),
            iaa.Fliplr(0.5, name="Fliplr"),  
            iaa.AdditiveGaussianNoise(scale=0.005 * 255),
            iaa.GaussianBlur(sigma=(1.0)),
            iaa.LinearContrast((0.5, 1.5), per_channel=0.5), 
            iaa.Affine(scale={"x": (0.5, 2), "y": (0.5, 2)}),
            iaa.Affine(rotate=(-40, 40)),
            iaa.Affine(shear=(-16, 16)),
            iaa.PiecewiseAffine(scale=(0.008, 0.03)),
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
        ], random_order=True)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train_npz":
            slice_name = self.sample_list[idx]
            data_path = os.path.join(self.data_dir, slice_name)
            data = np.load(data_path)
            image, label = data['image'], data['label']
            image, label = augment_seg(self.img_aug, image, label)
            x, y, _ = image.shape

            if x != self.img_size or y != self.img_size:
                image = cv.resize(image, (self.img_size, self.img_size), interpolation=cv.INTER_CUBIC)  
                label = cv.resize(label, (self.img_size, self.img_size), interpolation=cv.INTER_NEAREST)  
        else:
            slice_name = self.sample_list[idx]
            data_path = os.path.join(self.data_dir, slice_name)
            data = np.load(data_path)
            image, label = data['image'], data['label'] 
            x, y, _ = image.shape

            if x != self.img_size or y != self.img_size:
                image = cv.resize(image, (self.img_size, self.img_size), interpolation=cv.INTER_CUBIC) 
                label = cv.resize(label, (self.img_size, self.img_size), interpolation=cv.INTER_NEAREST)

        sample = {'image': image, 'label': label, 'name': slice_name.strip('.pnz')}
        if self.norm_x_transform is not None:
            sample['image'] = self.norm_x_transform(sample['image'].copy())
        if self.norm_y_transform is not None:
            sample['label'] = self.norm_y_transform(sample['label'].copy())
        return sample


class ISIC2018Dataset(Dataset):
    def __init__(self, data_root, target_size=(256, 256), mode='train'):
        super().__init__()
        self.mode = mode
        self.target_size = target_size
        self.file_list = glob.glob(os.path.join(data_root, "*.npz"))

        if len(self.file_list) == 0:
            raise ValueError(f"No .npz files found in {data_root}")

    def __len__(self):
        return len(self.file_list)

    def augment(self, image, label):

        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k)
            label = np.rot90(label, k)

        return image, label

    def __getitem__(self, idx):
        npz_path = self.file_list[idx]
        data = np.load(npz_path)
        image = data['image']
        label = data['label']
        if image.ndim == 2: 
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label.astype(np.uint8), (self.target_size[1], self.target_size[0]),
                           interpolation=cv2.INTER_NEAREST)
        if self.mode == 'train':
            image, label = self.augment(image, label)
        image = image.astype(np.float32)
        if image.max() > 2.0:  
            image = image / 255.0
        label = (label > 0).astype(np.float32)

        #image = image.astype(np.float32) / 255.0
        label = (label > 0).astype(np.float32)
        image_tensor = torch.from_numpy(image.transpose((2, 0, 1)))
        label_tensor = torch.from_numpy(label).unsqueeze(0)  # [1, H, W]

        return {'image': image_tensor, 'label': label_tensor, 'name': os.path.basename(npz_path)}

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
        self.beta = beta
        self.smooth = smooth

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
    def __init__(self, alpha=0.7, beta=0.3, gamma=4.0):
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta)
        self.gamma = gamma

    def forward(self, logits, labels):
        return torch.pow(self.tversky(logits, labels), self.gamma)


class SuperCombinedLoss(nn.Module):
    def __init__(self, ce_w=0.5, dice_w=0.5, ft_w=0.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.ft_loss = FocalTverskyLoss()
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
                if len(outputs) > 1: 
                    loss_main = loss_fn(outputs[0], labels)
                    labels_f = labels.float().unsqueeze(1) if labels.dim() == 3 else labels.float()
                    loss_aux1 = loss_fn(outputs[1], F.max_pool2d(labels_f, 2, 2).long())
                    loss_aux2 = loss_fn(outputs[2], F.max_pool2d(labels_f, 4, 4).long())
                    loss_aux3 = loss_fn(outputs[3], F.max_pool2d(labels_f, 8, 8).long())
                    total_loss = 1.0 * loss_main + 0.6 * loss_aux1 + 0.4 * loss_aux2 + 0.2 * loss_aux3
                else:
                    total_loss = loss_fn(outputs[0], labels)
            else:
                total_loss = loss_fn(outputs, labels)

            loss_to_backward = total_loss / accumulation_steps
            scaler.scale(loss_to_backward).backward()
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
    dice_scores = []
    iou_scores = []
    ac_scores = []  
    pr_scores = []  
    se_scores = []  
    sp_scores = []  

    for i, batch in enumerate(tqdm(dataloader, desc="Validating")):
        images, labels = batch['image'].to(device), batch['label'].to(device)
        img_names = batch.get('name', [f"img_{i}"])
        outputs = model(images)

        labels_cpu = labels.cpu().squeeze(1) if labels.dim() == 4 else labels.cpu()

        if isinstance(outputs, (list, tuple)):
            if len(outputs) > 1:
                probs_list = [torch.softmax(o.cpu(), dim=1) for o in outputs]
                fused_probs = torch.zeros_like(probs_list[0])
                weights = [0.8, 0.1, 0.06, 0.04]
                for j, probs in enumerate(probs_list[:4]):
                    up_probs = F.interpolate(probs, size=labels_cpu.shape[-2:], mode='bilinear', align_corners=False)
                    fused_probs += weights[j] * up_probs
                preds = torch.argmax(fused_probs, dim=1)
            else:
                preds = torch.argmax(torch.softmax(outputs[0].cpu(), dim=1), dim=1)
        else:
            preds = torch.argmax(torch.softmax(outputs.cpu(), dim=1), dim=1)

        for b_idx in range(preds.shape[0]):
            pred_np = preds[b_idx].numpy()
            label_np = labels_cpu[b_idx].numpy()
            pred_post = np.zeros_like(pred_np, dtype=np.uint8)
            class_mask = (pred_np == 1)
            if class_mask.any():
                labeled_mask, num_features = label(class_mask)
                if num_features > 0:
                    sizes = np.bincount(labeled_mask.ravel())[1:]
                    largest_idx = sizes.argmax() + 1
                    clean_mask = (labeled_mask == largest_idx)
                    final_mask = binary_closing(clean_mask, structure=np.ones((5, 5)))
                    pred_post[final_mask] = 1

            pred_mask = (pred_post == 1)
            true_mask = (label_np == 1)

            TP = np.sum(pred_mask & true_mask)  
            TN = np.sum((~pred_mask) & (~true_mask))  
            FP = np.sum(pred_mask & (~true_mask))  
            FN = np.sum((~pred_mask) & true_mask) 

            acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
            se = TP / (TP + FN + 1e-8)
            sp = TN / (TN + FP + 1e-8)
            pr = TP / (TP + FP + 1e-8)

            ac_scores.append(acc)
            se_scores.append(se)
            sp_scores.append(sp)
            pr_scores.append(pr)
            intersection = TP
            union = np.sum(pred_mask) + np.sum(true_mask) - intersection

            if union > 0:
                iou = intersection / union
                iou_scores.append(iou)

            dice = 2. * intersection / (np.sum(pred_mask) + np.sum(true_mask) + 1e-8)
            dice_scores.append(dice)

            if output_folder and b_idx == 0 and (i % save_freq == 0):
                save_2d_png(images[b_idx], label_np, pred_post, output_folder, epoch, img_names[b_idx])

    return {
        "dice": np.mean(dice_scores) if dice_scores else 0.0,
        "miou": np.mean(iou_scores) if iou_scores else 0.0,
        "ac": np.mean(ac_scores) if ac_scores else 0.0,
        "pr": np.mean(pr_scores) if pr_scores else 0.0,
        "se": np.mean(se_scores) if se_scores else 0.0,
        "sp": np.mean(sp_scores) if sp_scores else 0.0
    }


def save_2d_png(img_tensor, lbl_np, pred_np, folder, ep, name):
    os.makedirs(folder, exist_ok=True)
    base_name = name.split('.')[0]
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

    img_min = img_np.min()
    img_max = img_np.max()
    if img_max > img_min:
        img_np = (img_np - img_min) / (img_max - img_min)
    img_np = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    lbl_img = (lbl_np * 255).astype(np.uint8)
    pred_img = (pred_np * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(folder, f"ep{ep:03d}_{base_name}_img.png"), img_bgr)
    cv2.imwrite(os.path.join(folder, f"ep{ep:03d}_{base_name}_gt.png"), lbl_img)
    cv2.imwrite(os.path.join(folder, f"ep{ep:03d}_{base_name}_pred.png"), pred_img)
    overlay = img_bgr.copy()
    contours_gt, _ = cv2.findContours(lbl_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_gt, -1, (0, 255, 0), 2)  
    contours_pred, _ = cv2.findContours(pred_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_pred, -1, (0, 0, 255), 2) 
    cv2.imwrite(os.path.join(folder, f"ep{ep:03d}_{base_name}_overlay.png"), overlay)

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
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    TRAIN_DATA_ROOT = '..'
    VAL_DATA_ROOT = '..'

    OUT_DIR = f"..."
    os.makedirs(OUT_DIR, exist_ok=True)

    logger = setup_logger(os.path.join(OUT_DIR, "train.log"))
    logger.info(f"Started {MODEL_TYPE} on {DEVICE}")
    BATCH_SIZE, NUM_WORKERS = 4, 4
    LR, EPOCHS, ACCUM = 1e-5, 100, 1 
    TARGET_SIZE = (256, 256)
    NUM_CLASSES = 2  
    IN_CHANNELS = 3 
    WARMUP_EPOCHS = 15

    train_dataset = ISIC2018Dataset(data_root=TRAIN_DATA_ROOT, target_size=TARGET_SIZE, mode='train')
    val_dataset = ISIC2018Dataset(data_root=VAL_DATA_ROOT, target_size=TARGET_SIZE, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=NUM_WORKERS)

    model = model(in_channels=IN_CHANNELS, out_channels=NUM_CLASSES, base_c=32).to(DEVICE)
    dummy_input = torch.randn(1, IN_CHANNELS, TARGET_SIZE[0], TARGET_SIZE[1]).to(DEVICE)
    try:
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        macs, params = clever_format([macs, params], "%.3f")
        logger.info(f"Model: {MODEL_TYPE} | Parameters: {params} | FLOPs (MACs): {macs}")
    except Exception as e:
        logger.warning(f"Failed to compute FLOPs: {e}")
        logger.info(f"Model: {MODEL_TYPE} | Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    loss_fn = SuperCombinedLoss(ce_w=0.5, dice_w=0.5).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return float(epoch + 1) / float(max(1, WARMUP_EPOCHS))
        progress = float(epoch - WARMUP_EPOCHS) / float(max(1, EPOCHS - WARMUP_EPOCHS))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    best_dice = 0.0

    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{EPOCHS} | LR: {scheduler.get_last_lr()[0]:.2e}")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE, scaler, ACCUM)
        scheduler.step()

        val_metrics = validate(model, val_loader, DEVICE, os.path.join(OUT_DIR, "vis"), epoch + 1)

        logger.info(
            f"Train Loss: {train_loss:.4f} | Lesion Dice: {val_metrics['dice']:.4f} | "
            f"Lesion IoU: {val_metrics['miou']:.4f} | AC: {val_metrics['ac']:.4f} | "
            f"PR:{val_metrics['pr']:.4f} | SP: {val_metrics['sp']:.4f} | "
            f"SE:{val_metrics['se']:.4f} "
        )

        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best_model.pth"))
            logger.info(f" -> Best Model Saved! (Lesion Dice: {best_dice:.4f})")

if __name__ == '__main__':
    main()
