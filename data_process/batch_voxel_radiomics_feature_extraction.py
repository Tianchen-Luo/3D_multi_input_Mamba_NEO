# -*- coding: utf-8 -*-
import os
import nrrd
import SimpleITK as sitk
from tqdm import tqdm
from radiomics import featureextractor
from data_preprocess import preprocess_medical_image_remove_nan_inf, replace_inf_nan_for_array, create_folder, load_list_from_json


# Set up feature extractor configuration
def get_feature_extractor(using_yaml=True, param_path='../radiomics_params/Pyradiomics_Params_CT_ESO_pCR.yaml', params={'featureClass': {'original': None}}):
    if using_yaml:
        return featureextractor.RadiomicsFeatureExtractor(param_path)
    else:
        return featureextractor.RadiomicsFeatureExtractor(**params)


def extract_single_voxel_radiomics(image_path, mask_path, output_folder, extractor):
    """
    Extract voxel-based radiomics features from a medical image and mask.

    Parameters:
    - image_path (str): Path to the input image file.
    - mask_path (str): Path to the input segmentation mask file.
    - output_folder (str): Folder to save extracted features.
    - extractor (RadiomicsFeatureExtractor): Configured pyradiomics feature extractor.
    """
    image = preprocess_medical_image_remove_nan_inf(image_path)
    mask = preprocess_medical_image_remove_nan_inf(mask_path)

    result = extractor.execute(image, mask, voxelBased=True)

    for feature_name, feature_value in result.items():
        if isinstance(feature_value, sitk.Image):
            np_feature = sitk.GetArrayFromImage(feature_value)
            np_feature = replace_inf_nan_for_array(np_feature)

            output_path = os.path.join(output_folder, f'{feature_name}.nrrd')
            nrrd.write(output_path, np_feature)


def process_dataset(index_list, data_folder, output_folder, extractor):
    """
    Process a dataset of medical images and segmentation masks.

    Parameters:
    - index_list (list): List of image indices to process.
    - data_folder (str): Path to folder containing input image and mask files.
    - output_folder (str): Path to folder to save extracted features.
    - extractor (RadiomicsFeatureExtractor): Configured pyradiomics feature extractor.
    """
    for index in tqdm(index_list, desc="Processing images"):
        img_path = os.path.join(data_folder, f'{index}_img.nrrd')
        seg_path = os.path.join(data_folder, f'{index}_seg.nrrd')
        sub_folder_path = os.path.join(output_folder, index)

        if not os.path.exists(sub_folder_path) or len(os.listdir(sub_folder_path)) == 0:
            create_folder(sub_folder_path)
            extract_single_voxel_radiomics(img_path, seg_path, sub_folder_path, extractor)


def main():
 
    
    param_path = '../radiomics_params/Pyradiomics_Params_CT_ESO_pCR.yaml'
    using_yaml = True

    index_list = load_list_from_json('../json/abnormal_data_index/seg_shape_match_image.json')

    data_folder_path = '../resampled_1mm_data'
    voxel_features_saving_folder = '../voxel_radiomics_features'

    extractor = get_feature_extractor(using_yaml, param_path)

    process_dataset(index_list, data_folder_path, voxel_features_saving_folder, extractor)


if __name__ == "__main__":
    main()
