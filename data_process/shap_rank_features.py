import torch.nn as nn
import numpy as np
import shap
from data_preprocess import get_batch_from_dataloader


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).clone()


def calculate_shap_values(train_dataloader, model, device, batch_size, train_loader_len):
    model_wrapped = ModelWrapper(model)
    index_sum_list = [0] * 7

    name_list = [
        'CT',
        'original_gldm_LargeDependenceHighGrayLevelEmphasis.nrrd',
        'original_gldm_LowGrayLevelEmphasis.nrrd',
        'original_glrlm_LongRunLowGrayLevelEmphasis.nrrd',
        'original_glrlm_LowGrayLevelRunEmphasis.nrrd',
        'original_glszm_LargeAreaLowGrayLevelEmphasis.nrrd',
        'original_glszm_SmallAreaLowGrayLevelEmphasis.nrrd'
    ]

    for batch_index in range(train_loader_len):
        images, _ = get_batch_from_dataloader(train_dataloader, batch_index)
        background = images.to(device)

        g = shap.GradientExplainer(model_wrapped, background.clone())
        shap_values_batch3d = g.shap_values(background, nsamples=10)

        for num_index in range(batch_size):
            for image_index in range(len(index_sum_list)):
                index_sum_list[image_index] += np.sum(np.abs(shap_values_batch3d[num_index][image_index]))

    return name_list, index_sum_list


def get_sorted_feature_importance(name_list, index_sum_list):
    sorted_list = sorted(zip(name_list, index_sum_list), key=lambda x: x[1], reverse=True)
    sorted_names = [name for name, _ in sorted_list]
    return sorted_names


def get_feature_percentage(index_sum_list):
    total_sum = sum(index_sum_list)
    percentages_str = [f"{(value / total_sum) * 100:.6f}%" for value in index_sum_list]
    return percentages_str



def print_percentages(train_dataloader, model, device, batch_size, train_loader_len):
    name_list, index_sum_list = calculate_shap_values(train_dataloader, model, device, batch_size, train_loader_len)
    sorted_names = get_sorted_feature_importance(name_list, index_sum_list)
    percentages_str = get_feature_percentage(index_sum_list)

    print("Sorted Feature Names:", sorted_names)
    print("Feature Importance Percentages:", percentages_str)


