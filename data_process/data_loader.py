import torch
from torch.utils.data import Dataset
import h5py


class NEO_voxel_dataset(Dataset):
    def __init__(self, root_folder, split_file_path, selected_indices, transforms=None):
     
        self.root_folder = root_folder
        self.selected_indices = selected_indices
        self.transforms = transforms

        # Load split file and prepare file paths
        with open(split_file_path, 'r') as file:
            self.file_list = [f"{root_folder}/{line.strip()}.h5" for line in file]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        with h5py.File(self.file_list[idx], 'r') as file:
            selected_data = file['image'][self.selected_indices, :, :, :]   
            label = file['label'][()]  

 
        if self.transforms:
            selected_data = [self.transforms(torch.tensor(d, dtype=torch.float)) for d in selected_data]
        else:
            selected_data = [torch.tensor(d, dtype=torch.float) for d in selected_data]

 
        selected_data = torch.stack(selected_data)

  
        return selected_data, torch.tensor(label, dtype=torch.long)

    def check_h5_contents(self, file_path):
    
        with h5py.File(file_path, 'r') as file:
            print(f"Datasets in {file_path}: {list(file.keys())}")
