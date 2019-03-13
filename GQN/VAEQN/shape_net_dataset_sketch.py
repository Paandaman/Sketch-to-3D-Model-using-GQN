import os
import torch
from torch.utils.data import Dataset

class ShapeNetDatasetSketch(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        scene_path = os.path.join(self.root_dir, "{}.pt".format(idx))
        data = torch.load(scene_path)
        return data