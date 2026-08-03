import SimpleITK as sitk
import os
from glob import glob
from tqdm import tqdm
import numpy as np


class SynapsePreprocessor:
    def __init__(self, target_spacing=(1.5, 1.5, 2.0)):
        self.target_spacing = target_spacing
        self.resampler = sitk.ResampleImageFilter()
        self.resampler.SetOutputDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        self.resampler.SetOutputSpacing(self.target_spacing)

    def resample_image(self, sitk_image, is_label):
        original_spacing = sitk_image.GetSpacing()
        original_size = sitk_image.GetSize()

        new_size = [int(round(osz * ospc / tspc)) for osz, ospc, tspc in
                    zip(original_size, original_spacing, self.target_spacing)]

        self.resampler.SetSize(new_size)
        self.resampler.SetOutputOrigin(sitk_image.GetOrigin())

        if is_label:
            self.resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            self.resampler.SetInterpolator(sitk.sitkBSpline)

        return self.resampler.Execute(sitk_image)

    def clip_and_normalize_hu(self, sitk_image, lower_bound=-125, upper_bound=275):
        np_image = sitk.GetArrayFromImage(sitk_image)
        np_image = np.clip(np_image, lower_bound, upper_bound)
        np_image = (np_image - lower_bound) / (upper_bound - lower_bound)

        # Volver a crear la imagen SimpleITK con los metadatos originales
        new_sitk_image = sitk.GetImageFromArray(np_image)
        new_sitk_image.CopyInformation(sitk_image)
        return new_sitk_image


def preprocess_synapse_dataset(original_data_path, preprocessed_data_path):
    print(f"Starting Synapse preprocessing from '{original_data_path}' to '{preprocessed_data_path}'")

    image_files = sorted(glob(os.path.join(original_data_path, 'train_image', '*.nii.gz')))
    label_files = sorted(glob(os.path.join(original_data_path, 'train_labels', '*.nii.gz')))
    target_img_folder = os.path.join(preprocessed_data_path, 'imagesTr')
    target_lbl_folder = os.path.join(preprocessed_data_path, 'labelsTr')
    os.makedirs(target_img_folder, exist_ok=True)
    os.makedirs(target_lbl_folder, exist_ok=True)

    preprocessor = SynapsePreprocessor()

    for img_path, lbl_path in tqdm(zip(image_files, label_files), total=len(image_files), desc="Preprocessing Synapse"):
        try:
            img_filename = os.path.basename(img_path)
            lbl_filename = img_filename.replace('.nii.gz', '_seg.nii.gz')
            assert os.path.basename(lbl_path) == lbl_filename

            img_sitk = sitk.ReadImage(img_path, sitk.sitkFloat32)  # Cargar como float para normalización
            lbl_sitk = sitk.ReadImage(lbl_path)
            img_sitk = preprocessor.clip_and_normalize_hu(img_sitk)
            resampled_img_sitk = preprocessor.resample_image(img_sitk, is_label=False)
            resampled_lbl_sitk = preprocessor.resample_image(lbl_sitk, is_label=True)

            sitk.WriteImage(resampled_img_sitk, os.path.join(target_img_folder, img_filename))
            sitk.WriteImage(resampled_lbl_sitk, os.path.join(target_lbl_folder, lbl_filename))

        except Exception as e:
            print(f"Could not process pair {img_path} and {lbl_path}. Error: {e}")


if __name__ == '__main__':
    ORIGINAL_SYNAPSE_PATH = "...."
    PREPROCESSED_SYNAPSE_PATH = "....."

    preprocess_synapse_dataset(ORIGINAL_SYNAPSE_PATH, PREPROCESSED_SYNAPSE_PATH)
    print("Synapse preprocessing finished!")




