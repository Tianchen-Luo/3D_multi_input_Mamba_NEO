import nrrd
import SimpleITK as sitk
import numpy as np
import os
import json

def replace_inf_nan_for_array(array, value=0):
 
    array = np.nan_to_num(array, nan=value, posinf=value, neginf=value)
    return array

def preprocess_medical_image_remove_nan_inf(image_path):

    image = sitk.ReadImage(image_path)
    np_image = sitk.GetArrayFromImage(image)
    np_image = np.nan_to_num(np_image)  
    return sitk.GetImageFromArray(np_image)


def process_nrrd(file_path, min_threshold, max_threshold, save_path = None):
    
    data, header = nrrd.read(file_path)

    data[data < min_threshold] = min_threshold
    data[data > max_threshold] = max_threshold

    if save_path:

        nrrd.write(save_path, data, header)
    else:
        return data
    

def get_bounding_box(mask):

    non_zero_indices = np.argwhere(mask)
    upper_left = non_zero_indices.min(axis=0)
    lower_right = non_zero_indices.max(axis=0)
    return [upper_left[2], lower_right[2], upper_left[1], lower_right[1], upper_left[0], lower_right[0]]

def crop_image(image, bbox):
    return image[bbox[4]:bbox[5]+1, bbox[2]:bbox[3]+1, bbox[0]:bbox[1]+1]

def pad_image(image, target_size):
    padding = [(0, max(0, target_size[i] - image.shape[i])) for i in range(3)]
    return np.pad(image, padding, mode='constant')



def load_nrrd(file_path):
    
    try:
        data, header = nrrd.read(file_path)
        return data, header
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None, None
    
def sitk_image_to_array(sitk_image):
  
    return sitk.GetArrayFromImage(sitk_image)

def array_to_sitk_image(numpy_array):
 
    return sitk.GetImageFromArray(numpy_array)


def transpose_image_to_xyz(image):
    return sitk.PermuteAxes(image, [0,2,1])


def save_image(image, file_path):
    sitk.WriteImage(image, file_path)

def resize_image(image, target_size, is_segmentation=False):

    resample = sitk.ResampleImageFilter()
    resample.SetSize(target_size)

    if is_segmentation:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(image)

def pad_image(image, target_size, padding_value=-120):

    current_size = image.GetSize()
    size_diff = [max(0, ts - cs) for ts, cs in zip(target_size, current_size)]

    padding = [(diff // 2, diff - diff // 2) for diff in size_diff]
    if len(padding) < 3:
        padding += [(0, 0)] * (3 - len(padding))

    lower_padding, upper_padding = zip(*padding)

    return sitk.ConstantPad(image, lower_padding, upper_padding, padding_value)



def resize_and_pad_image(image, target_size, is_segmentation=False, padding_value=0):

    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    scaling_factor = [float(osz)/ts for osz, ts in zip(original_size, target_size)]
    new_spacing = [osp*sf for osp, sf in zip(original_spacing, scaling_factor)]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(new_spacing)
    
    if is_segmentation:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)

    resized_image = resampler.Execute(image)

    current_size = resized_image.GetSize()
    pad_size = [max(0, ts - cs) for ts, cs in zip(target_size, current_size)]
    padding = [(size // 2, size - size // 2) for size in pad_size]

    while len(padding) < 3:
        padding.append((0, 0))

    lower_padding, upper_padding = zip(*padding)
    
    return sitk.ConstantPad(resized_image, lower_padding, upper_padding, padding_value)


def process_single_image(image_path, target_size, is_exception, is_segmentation=False, padding_value=-120, transpose=False):

    image_array, cropped_header = load_nrrd(image_path)
    image = array_to_sitk_image(image_array)

    if is_exception:
        processed_image = resize_and_pad_image(image, target_size, is_segmentation, padding_value=padding_value)
    else:
        processed_image = pad_image(image, target_size, padding_value)

    processed_image_array = sitk_image_to_array(processed_image)
    
    if transpose:
        processed_image = transpose_image_to_xyz(processed_image)
    
    return processed_image

def process_images_in_batch(image_paths, segmentation_paths, target_size, exception_paths, output_folder, img_padding_value=-120, seg_padding_value = 0):

    for img_path, seg_path in zip(image_paths, segmentation_paths):

        basename =os.path.basename(img_path) 
        index = basename.split('.')[0].split('_')[0]
        
        print(f'Current processing index is:{index}')
        print(img_path)

        is_exception = basename in exception_paths

        print(f'Is it exception? {is_exception}')
        processed_image = process_single_image(img_path, target_size, is_exception,is_segmentation=False, padding_value=img_padding_value, transpose=False)
        processed_segmentation = process_single_image(seg_path, target_size, is_exception, is_segmentation=True, padding_value=seg_padding_value, transpose=False)

        processed_image_array = sitk_image_to_array(processed_image).transpose(2,1,0)
        processed_seg_array = sitk_image_to_array(processed_segmentation).transpose(2,1,0)

        nrrd.write(os.path.join(output_folder, basename),processed_image_array)
        nrrd.write(os.path.join(output_folder, f'{index}_seg.nrrd'),processed_seg_array)

def resample_nrrd(input_file_path, output_file_path, new_x_spacing=1.0, new_y_spacing=1.0):
    
    input_image = sitk.ReadImage(input_file_path)

    original_spacing = input_image.GetSpacing()
    original_size = input_image.GetSize()

    new_spacing = [new_x_spacing, new_y_spacing, original_spacing[2]]

    new_size = [int(round(osz*ospc/nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]

    resampled_image = sitk.Resample(input_image, new_size, sitk.Transform(), sitk.sitkLinear, 
                                    input_image.GetOrigin(), new_spacing, input_image.GetDirection(), 0, 
                                    input_image.GetPixelID())

    sitk.WriteImage(resampled_image, output_file_path)



def load_list_from_json(json_file_path):
    try:
        with open(json_file_path, 'r') as file:
            my_list = json.load(file)
        print("List loaded from JSON file:", my_list)
        return my_list
    except Exception as e:
        print("An error occurred while loading the JSON file:", e)
        return None
    
def create_folder(folder_path):
   
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def list_sub_files_full_path(folder_path):
 
    sub_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            sub_files.append(os.path.join(root, file))

    return sub_files


def get_batch_from_dataloader(dataloader, index):
    """
    Retrieve the batch that contains the specified index from the dataloader.
    Note: This function assumes that the dataloader shuffles data,
    so the index refers to the position within the batch rather than the dataset.
    
    Parameters:
    - dataloader: The PyTorch DataLoader instance.
    - index: The index of the data point you want to retrieve.
    
    Returns:
    - The batch that contains the specified index. This includes both images and labels.
    """
    batch_index = index // dataloader.batch_size
    current_index = 0
    for batch in dataloader:
        if current_index == batch_index:
            return batch
        current_index += 1

def apply_mask(ct_image, roi_segmentation, replace_value = 0, new_value = 0):
    """
    Apply the ROI segmentation to the CT image.
    
    Parameters:
    ct_image (np.ndarray): 3D numpy array representing the CT image.
    roi_segmentation (np.ndarray): 3D numpy array representing the ROI segmentation.
    
    Returns:
    np.ndarray: 3D numpy array with CT image values only in the ROI area, other areas set to -120.
    """
    if ct_image.shape != roi_segmentation.shape:
        raise ValueError("The CT image and ROI segmentation must have the same shape.")

    # Create a copy of the CT image to avoid modifying the original
    ct_with_roi = np.copy(ct_image)

    # Set values outside the ROI to -120
    ct_with_roi[roi_segmentation == replace_value] = new_value
    
    return ct_with_roi
