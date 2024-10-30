# Voxel-level Radiomics and Deep Learning for Predicting Pathologic Complete Response in Esophageal Squamous Cell Carcinoma after Neoadjuvant Immunotherapy and Chemotherapy

## Overview
This is the primary code repository for the research: Voxel-level Radiomics and Deep Learning for Predicting Pathologic Complete Response in Esophageal Squamous Cell Carcinoma after Neoadjuvant Immunotherapy and Chemotherapy
Our deep learning model: 3D_multi_input_Mamba was modified based on: https://pypi.org/project/mamba-ssm/

## Usage
### **1.Install Python 3.10 and you can execute the following command to install required packages.**

```bash
pip install -r requirements.txt
```

#### Other requirements:
- Linux
- NVIDIA GPU
- PyTorch 1.12+
- CUDA 11.8

### **2. Before training the Multi-inputs 3D-Mamba model, process the data, extracting Voxel-level Radiomics feature maps and make dataloader first**
   
#### **data_process/data_preprocess.py:**
  
This is a utils file that contains all Python definitions related to image preprocessing, including operations such as threshold clipping, cropping, resampling, resizing, and adding padding to the original CT images.
Other scripts will import from here.
  
#### **data_process/batch_voxel_radiomics_feature_extraction.py:**

This script extracts voxel-level radiomics feature maps based on CT images and their corresponding ROIs after applying the same preprocessing operations. Before running the script, a radiomics configuration file (.yaml), the data folder containing the CT images, and the target folder path for saving the extracted features must be provided.

  
#### **data_process/make_dataset.py:**
This script is used to combine and generate .h5 files from the selected feature maps and CT images for model training.

  
#### **data_process/data_loader.py:**
This script is used to create a dataloader for reading .h5 files.

#### **data_process/shap_rank_features.py:**
After the model training is completed, this script calculates the impact ratio of different input voxel-level SHAP values on the model's decisions.
  
#### **data_process/shap_plot.py:**
Overlay the voxel-level SHAP feature maps with the original images for visualization and analyze the model's decisions in combination with clinical judgment.

### **3. Models and training**

#### **model/multi_input_vision_mamba.py:**
The code for the Multi-inputs 3D-Mamba model requires a Linux system and the installation of the mamba_ssm==1.2.0 package.

#### **model/transforms.py:**
The data augmentation code includes operations such as adding Gaussian noise, Random Affine transformations, and Horizontal Flip.

#### **main.py:**
This is the model training code, which continuously updates and saves the best-performing model on the validation set during the training process.


## Contact
If you have any questions or suggestions, feel free to contact us:
- **Zhen Zhang** , [Zhen.zhang@maastro.nl](mailto:Zhen.zhang@maastro.nl)
- **Tianchen Luo** , [luotianchen218@163.com](mailto:luotianchen218@163.com)

