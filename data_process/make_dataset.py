import os
import sys
import h5py
import numpy as np
from tqdm import tqdm
from scipy.ndimage import zoom
from data_preprocess import create_folder, list_sub_files_full_path
sys.path.append("..")


def resize_3d_image(volume, target_shape):
    """
    Resize a 3D volume using cubic interpolation.

    Parameters:
    - volume: numpy.ndarray, the 3D volume to resize, expected shape is (depth, height, width).
    - target_shape: tuple, the target shape as (target_depth, target_height, target_width).

    Returns:
    - resized_volume: numpy.ndarray, the resized 3D volume with shape `target_shape`.
    """
    zoom_factors = [n / o for n, o in zip(target_shape, volume.shape)]
    resized_volume = zoom(volume, zoom_factors, order=3)
    return resized_volume


def batch_resize_h5_files(folder_path, target_shape=(32, 64, 64)):
    """
    Batch resize 3D images in all .h5 files within a folder.

    Parameters:
    - folder_path: str, the path to the folder containing .h5 files.
    - target_shape: tuple, the target shape for 3D images, default is (32, 64, 64).
    """
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.h5'):
            file_path = os.path.join(folder_path, file_name)
            with h5py.File(file_path, 'r+') as h5_file:
                data = h5_file['image'][:]
                resized_images = np.zeros((data.shape[0],) + target_shape)
                for i, image in enumerate(data):
                    resized_images[i] = resize_3d_image(image, target_shape)
                del h5_file['image']
                h5_file.create_dataset('image', data=resized_images, compression="gzip")
                print(f'Resized {file_name}')


def merge_voxel_and_baseline_datasets(baseline_folder, voxel_folder_path, output_folder, target_shape=(32, 64, 64)):
    """
    Merge voxel-based features and baseline data into a combined dataset.

    Parameters:
    - baseline_folder: str, path to the folder containing baseline .h5 files.
    - voxel_folder_path: str, path to the folder containing voxel .h5 files.
    - output_folder: str, path to the folder to save the merged datasets.
    - target_shape: tuple, the target shape for resizing baseline images.
    """
    create_folder(output_folder)
    baseline_list = sorted(list_sub_files_full_path(baseline_folder))

    for baseline_path in tqdm(baseline_list, desc="Merging datasets"):
        basename = os.path.basename(baseline_path)
        voxel_path = os.path.join(voxel_folder_path, basename)

        with h5py.File(baseline_path, 'r') as h5f_baseline, h5py.File(voxel_path, 'r') as h5f_voxel:
            image_baseline = h5f_baseline['image'][:]
            image_voxels = h5f_voxel['image'][:]
            label = h5f_voxel['label'][:]

            image_baseline_reorder = np.transpose(image_baseline, (0, 3, 1, 2))
            image_reshape = resize_3d_image(image_baseline_reorder[0], target_shape)
            image1_expanded = np.expand_dims(image_reshape, axis=0)

            new_image = np.concatenate((image1_expanded, image_voxels), axis=0)

            save_path = os.path.join(output_folder, basename)
            with h5py.File(save_path, 'w') as f:
                f.create_dataset('image', data=new_image, compression="gzip")
                f.create_dataset('label', data=[int(label)], compression="gzip")
                print(f'Merged {basename}')


def main():
    """
    Main function for resizing datasets and merging voxel and baseline data.
    """
    # Set up directories
    baseline_folder = '../baseline_h5py_dataset/'
    voxel_folder_path = '../resized_radiomics_h5py_dataset'
    output_folder = '../voxel_and_baseline_h5py_dataset'

    # Batch resize h5 files
    batch_resize_h5_files(voxel_folder_path)

    # Merge voxel and baseline datasets
    merge_voxel_and_baseline_datasets(baseline_folder, voxel_folder_path, output_folder)


if __name__ == "__main__":
    main()
