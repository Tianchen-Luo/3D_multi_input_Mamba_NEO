import torch
import torch.nn as nn
import shap
import os
import numpy as np
import SimpleITK as sitk
import h5py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_preprocess import load_nrrd, process_single_image, apply_mask
from shap_rank_features import ModelWrapper


def initialize_model(model_checkpoint, device, model_class, pos_weight_val):
    model = model_class().to(device)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.load_state_dict(torch.load(model_checkpoint))
    return model, criterion


def load_data_with_index(index, folder_path, selected_indice):
    test_path = os.path.join(folder_path, f'{index}.h5')
    h5f = h5py.File(test_path, 'r')
    image = h5f['image'][:]
    selected_images = image[selected_indice, :, :, :]
    return torch.tensor(selected_images, dtype=torch.float)


def load_shap_single_input(index_list, folder_path, selected_indice):
    test_image = load_data_with_index(index_list, folder_path, selected_indice)
    return test_image.unsqueeze(1)


def load_original_image_and_seg(index, target_shape, img_padding_value, seg_padding_value, cropped_tumor_path, cropped_tumor_seg_path):
    cropped_tumor_path = f'{cropped_tumor_path}/{index}_img.nrrd'
    cropped_tumor_seg_path = f'{cropped_tumor_seg_path}/{index}_seg.nrrd'
    
    img_array, img_header = load_nrrd(cropped_tumor_path)
    current_shape = img_array.shape
    is_exception = False

    if current_shape[0] > target_shape[2] or current_shape[1] > target_shape[0] or current_shape[2] > target_shape[1]:
        is_exception = True
    
    processed_segmentation = sitk.GetArrayFromImage(process_single_image(cropped_tumor_seg_path, target_shape, is_exception, is_segmentation=True, padding_value=seg_padding_value, transpose=False))
    processed_image = sitk.GetArrayFromImage(process_single_image(cropped_tumor_path, target_shape, is_exception,is_segmentation=False, padding_value=img_padding_value, transpose=False))
    return processed_image, processed_segmentation


def load_shap_use_torch(choose_index, index_list, voxel_and_baseline_h5py_dataset_path, selected_indice):
    test_image = load_data_with_index(index_list[2*choose_index], voxel_and_baseline_h5py_dataset_path, selected_indice)
    comparison_image = load_data_with_index(index_list[2*choose_index+1], voxel_and_baseline_h5py_dataset_path, selected_indice)
    input_tensor = torch.stack([test_image, comparison_image], dim=0)
    return test_image, comparison_image, input_tensor, index_list[2*choose_index], index_list[2*choose_index+1]


def plot_two_images(array1, array2, title1, title2, save_path=None):
    array1_transformed = np.fliplr(np.rot90(np.rot90(array1, 2), 3))
    array2_transformed = np.fliplr(np.rot90(np.rot90(array2, 2), 3))

    fig = make_subplots(rows=1, cols=2, subplot_titles=(title1, title2), horizontal_spacing=0.05)
    fig.add_trace(go.Heatmap(z=array1_transformed, colorscale='gray', showscale=True), row=1, col=1)

    color_scale = [[0.0, "blue"], [0.5, "white"], [1.0, "red"]]
    fig.add_trace(go.Heatmap(z=array2_transformed, colorscale=color_scale, showscale=True), row=1, col=2)

    fig.update_xaxes(showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(showticklabels=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(width=800, height=400, margin=dict(l=20, r=20, t=50, b=20))

    if save_path:
        fig.write_image(save_path)
    else:
        fig.show()


def main(model, model_checkpoint, voxel_and_baseline_h5py_dataset_path, target_shape, img_padding_value, seg_padding_value, selected_indice, choose_index, index_list, cropped_tumor_path, cropped_tumor_seg_path, pos_weight_val):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = initialize_model(model_checkpoint, device, model, pos_weight_val)
    model_wrapped = ModelWrapper(model)

    test_image, comparison_image, input_tensor, index, comparison_index = load_shap_use_torch(choose_index, index_list, voxel_and_baseline_h5py_dataset_path, selected_indice)
    img_array = test_image.numpy()[0]

    processed_image, processed_segmentation = load_original_image_and_seg(index, target_shape, img_padding_value, seg_padding_value, cropped_tumor_path, cropped_tumor_seg_path)

    background = input_tensor.to(device)
    g = shap.GradientExplainer(model_wrapped, background.clone())
    shap_values_batch3d = g.shap_values(background, nsamples=20)
    shap_values_array_img1 = shap_values_batch3d[0][0]
    shap_values_array_img1 = apply_mask(shap_values_array_img1, processed_segmentation)

    plot_two_images(img_array[0], shap_values_array_img1[0], title1="CT", title2="Feature map")


if __name__ == "__main__":
    model_checkpoint = 'check_points/masked_baseline_mamba_single_input_[0]/model_epoch82_best_auc_0.56.pth'
    voxel_and_baseline_h5py_dataset_path = 'voxel_and_baseline_h5py_dataset_remake_with_mask'
    image_shape = [64, 64, 32]
    img_padding_value = -120
    seg_padding_value = 0
    selected_indice = [0]
    choose_index = 20
    index_list = ['dataset_files/training_index.txt']
    cropped_tumor_path = 'cropped_tumor_folder'
    cropped_tumor_seg_path = 'cropped_tumor_folder'
    pos_weight_val = 4.0
    
    main(model_checkpoint, voxel_and_baseline_h5py_dataset_path, image_shape, img_padding_value, seg_padding_value, selected_indice, choose_index, index_list, cropped_tumor_path, cropped_tumor_seg_path, pos_weight_val)
