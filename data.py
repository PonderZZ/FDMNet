import torch
import os
import numpy as np
import random
from torch.utils.data import Dataset
import h5py
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob



class MedicalImageDataset(Dataset):
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        raise NotImplementedError

def preprocess_ct_volume(volume_np, w_level=40, w_width=400):
    lower_bound = w_level - w_width / 2
    upper_bound = w_level + w_width / 2
    volume_np = np.clip(volume_np, lower_bound, upper_bound)
    volume_np = (volume_np - lower_bound) / (w_width + 1e-8)
    return volume_np


def get_train_transforms(img_size=(224, 224)):
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1], always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.7),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.7),
        A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.RandomGamma(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(p=0.3),
        A.GaussNoise(p=0.3),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=(224, 224)):
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1], always_apply=True),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])


class SynapseUnified2dDataset(MedicalImageDataset):
    def __init__(self, base_dir, patient_ids, transforms, mode='train', train_data_format='npz'):
        self.base_dir = base_dir
        self.patient_ids = patient_ids
        self.transforms = transforms
        self.mode = mode
        self.train_data_format = train_data_format

        self.sample_list = []
        if self.mode == 'train':
            if self.train_data_format == 'npz':
                data_dir = os.path.join(base_dir, 'train_npz')
                all_slices = os.listdir(data_dir)
                for pid in self.patient_ids:
                    patient_slices = [s.replace('.npz', '') for s in all_slices if s.startswith(pid)]
                    self.sample_list.extend(patient_slices)
                random.shuffle(self.sample_list)
                print(f"Initialized Unified Dataset in TRAIN (npz) mode: {len(self.sample_list)} slices.")
            else:
                self.sample_list = self.patient_ids
                print(
                    f"Initialized Unified Dataset in TRAIN (h5) mode: {len(self.sample_list)} volumes (will be sliced).")

        elif self.mode == 'validation':
            self.sample_list = self.patient_ids
            print(f"Initialized Unified Dataset in VALIDATION (h5) mode: {len(self.sample_list)} volumes.")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.mode == 'train':
            slice_name = self.sample_list[idx]
            data_path = os.path.join(self.base_dir, 'train_npz', slice_name + '.npz')
            try:
                data = np.load(data_path)
                image, label = data['image'], data['label']
            except Exception as e:
                raise RuntimeError(f"Error loading training slice {data_path}: {e}")

            image_processed = preprocess_ct_volume(image)

            transformed = self.transforms(image=np.expand_dims(image_processed, axis=-1), mask=label)

            image_tensor = transformed['image']
            label_tensor = transformed['mask'].long()

            return {'image': image_tensor, 'label': label_tensor}

        elif self.mode == 'validation':
            case_name = self.sample_list[idx]
            data_path = os.path.join(self.base_dir, 'test_vol_h5', case_name + '.npy.h5')
            try:
                with h5py.File(data_path, 'r') as hf:
                    image_vol, label_vol = hf['image'][:], hf['label'][:]
            except Exception as e:
                raise RuntimeError(f"Error loading validation volume {data_path}: {e}")

            image_vol_processed = preprocess_ct_volume(image_vol)

            image_tensor_slices = []
            label_tensor_slices = []

            for i in range(image_vol_processed.shape[0]):
                transformed = self.transforms(
                    image=np.expand_dims(image_vol_processed[i], axis=-1),
                    mask=label_vol[i]
                )
                image_tensor_slices.append(transformed['image'])
                label_tensor_slices.append(transformed['mask'].long())

            image_tensor_vol = torch.stack(image_tensor_slices, dim=0)
            label_tensor_vol = torch.stack(label_tensor_slices, dim=0)

            return {'image': image_tensor_vol, 'label': label_tensor_vol, 'case_name': case_name}
