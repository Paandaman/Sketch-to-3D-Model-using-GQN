import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import glob
import matplotlib.pyplot as plt

class Shapenet_dataset(Dataset):
    # Loads data from the ShapeNet dataset saved in multi view representation
    # in tfrecords format and converts it to pytorch tensors

    def __init__(self, root_dir_sketch, root_dir_real, batch_size):

        self.root = root_dir_sketch
        self.file_names = self.get_names(root_dir_sketch)
        self.file_names_real = root_dir_real
        self.pos_dim = 5 
        self.seq_length = 15
        self.image_size = 64
        self.batch_size = batch_size
        self.toTensor = torchvision.transforms.ToTensor()
        self.reduceSize = torchvision.transforms.Resize([64, 64])


    def __len__(self):
        return len(self.get_names(self.root))

    def get_names(self, dir):
        file_names = []
        for name in glob.glob(dir+"/model*"):
            file_names.append(name)
        return file_names


    def __getitem__(self, idx):
        batch_names = self.file_names[idx:idx+self.batch_size]
        batch = self.load_data(batch_names)
        return batch

    def load_data(self, file_name):
        data = self.convert_record(file_name)
        return data

    def convert_record(self, record, batch_size=None):
        scenes, scenes_real = self.process_record(record, batch_size)
        
        return (scenes, scenes_real)

    def process_record(self, record, batch_size=None):
        
        batch_scenes = []
        batch_scenes_real = []
        for name in record:
            scenes = []
            scenes_real = []

            img_n = name.split("/")[-1]
            img_n_real = self.file_names_real + "/" + img_n
            for i in range(0, 15):
                model_nr_id_view = name + "/" + img_n + "_" + str(i) 
                model_nr_id_view_real = img_n_real + "/" + img_n + "_" + str(i) 

                img_name = model_nr_id_view + ".jpg"
                img_name_real = model_nr_id_view_real + ".jpg"

                label_name = model_nr_id_view + ".txt"
                label_name_real = model_nr_id_view_real + ".txt"

                image = Image.open(img_name)
                image_real = Image.open(img_name_real)

                image = self.reduceSize(image)
                image_real = self.reduceSize(image_real)

                image = self.process_images(image, False)
                image_real = self.process_images(image_real, True)

                file = open(label_name)
                file_real = open(label_name_real)

                labels = self.process_labels(file.read())
                labels_real = self.process_labels(file_real.read())

                file.close()
                file_real.close()

                scenes.append((image, labels))
                scenes_real.append((image_real, labels_real))

            batch_scenes.append(scenes)
            batch_scenes_real.append(scenes_real)

        return batch_scenes, batch_scenes_real

    def process_images(self, image, real):
        if real:
            image = np.array(image)
            image = torch.tensor(image).unsqueeze(0).type(torch.FloatTensor)
            OldMin = 0
            OldMax = 255
            NewMin = 0
            NewMax = 1
            OldRange = (OldMax - OldMin)  
            NewRange = (NewMax - NewMin)  
            image = torch.add(torch.div((image-OldMin)*NewRange, OldRange), NewMin)
            
        else:
            image = self.toTensor(image)

        return image

    def process_labels(self, labels):
        separate_labels = labels.split("_")
        num_labels = []
        for lab in separate_labels:
            num_labels.append(np.float(lab))
        to_tensor = torch.FloatTensor(num_labels)
        return to_tensor
