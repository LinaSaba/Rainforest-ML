import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
from glob import glob
import matplotlib.pyplot as plt
import numpy as np

def get_classes_list():
    """
    This function defines the classes we would use during the project.
    The class label will be predicted for a given example of input images.
    Returns:
        list of classes : list
        len of the list : str
    """
    classes = ['clear', 'cloudy', 'haze', 'partly_cloudy',
               'agriculture', 'artisinal_mine', 'bare_ground', 'blooming',
               'blow_down', 'conventional_mine', 'cultivation', 'habitation',
               'primary', 'road', 'selective_logging', 'slash_burn', 'water']
    return classes, len(classes)


class ChannelSelect(torch.nn.Module):
    """This class is to be used in transforms. Compose when you want to use selected channels. e.g only RGB.
    It works only for a tensor, not PIL object.

    Args:
        channels (list or int): The channels you want to select from the original image (4-channel).
    Returns:
        img
    """
    def __init__(self, channels=[0, 1, 2]):
        #The super function returns a temporary object of the superclass that allows access to all of its methods to its child class.
        super().__init__()
        self.channels = channels

    def forward(self, img):
        """
        Args:
            img (Tensor): Image
        Returns:
            Tensor: Selected channels from the image.
        """
        return img[self.channels, ...]

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RainforestDataset(Dataset):
    def __init__(self, root_dir, trvaltest, transform):
        """
        This function's goal is to define how we initialise the data
        At first we create two lists image_paths and split_labels to store the path labels of each image
        Second we binarise the multi-labels from the string
        Third we make the fit_transform of the multi-labels which will be stored to be used for the train test
        Last we define the train and test split if it s 0 then train else it s 1
        """
        self.image_paths = []
        self.split_labels = []
        self.transform = transform
        #https://www.w3schools.com/python/pandas/pandas_csv.asp
        #https://datatofish.com/convert-pandas-dataframe-to-list/
        root_cc =  root_dir+ "/train_v2.csv"
        root_csv = pd.read_csv(root_cc)
        #root_csv = root_csv.iloc[:100]
        liste = root_csv.values.tolist()
        for line in liste :
            image_path = root_dir + "/train-tif-v2/" +line[0]+".tif"
            self.image_paths.append(image_path)
            #https://www.w3schools.com/python/ref_string_split.asp
            self.split_labels.append(line[1].split())
        #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html
        classes,_ = get_classes_list()
        mlb = MultiLabelBinarizer(classes=classes)
        #https://scikit-learn.org/stable/modules/preprocessing.html
        y = mlb.fit_transform(self.split_labels)
        #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        X_train, X_test, y_train, y_test = train_test_split(self.image_paths, y, test_size=0.33, random_state=0)
        if trvaltest == 0 :
            self.image_paths = X_train
            self.split_labels = y_train
        else:
            self.image_paths = X_test
            self.split_labels = y_test


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        This function get the label and filename and load the image from file after transforming it to Tensor
        """
        #https://www.geeksforgeeks.org/python-pil-image-open-method/
        image = Image.open(self.image_paths[idx])
        label = self.split_labels[idx]
        #print(np.shape(label))
        #print(image)
        if self.transform:
            #https://pytorch.org/vision/stable/transforms.html
            image = self.transform(image)
        else:
            image = torch.transforms.ToTensor()(image)
        sample = {'image': image, 'label': label, 'filename': self.image_paths[idx]}
        return sample
