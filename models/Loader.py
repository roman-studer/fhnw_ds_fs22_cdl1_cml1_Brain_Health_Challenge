import torchio as tio
import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os

class MRIDataset(Dataset):
    """
    Implements __getitem__ method to get items for batches
    Attributes
    ----------
    image_path (string): Path to the image folder
    test_path (string: Path to where tabular data is stored
    annotation_path (string): path to annotation where label is stored
    transform (object, callable): Optional transform to be applied
        on a sample.
    labels: labels extracted from annotation_path
    images: iterable list of paths to all images (could also be images themselves)
    test_results: test results extracted from test_path
    ----------
    Methods
    ----------
    forward(images): takes images and does forward feed calculation
    """
    def __init__(self, image_path, test_path, annotation_path, transform=None):
        """
        Parameters:
            image_path (string): Path to the image folder
            test_path (string: Path to where tabular data is stored
            annotation_path (string): path to annotation where label is stored
            transform (object, callable): Optional transform to be applied
                on a sample.
        """
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.test_path = test_path
        self.transform = transform
        
        self.labels = pd.read_csv(self.annotation_path)
        self.images = os.listdir(self.image_path)
        self.test_results = pd.read_csv(self.test_path)

    def __len__(self):
        """
        returns amount of images in total
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Parameters:
            idx (int): index to image
        """
        # fixme should be handled in another way. maybe in configuration file?
        idx_to_label = {
            'CN': 0,
            'MCI': 1,
            'AD': 2
        }

        test_results = self.test_results[["CDMEMORY", "CDORIENT", "CDJUDGE", "CDCOMMUN", "CDHOME", "CDCARE", "CDGLOBAL"]].sample(n=1)
        test_results = torch.from_numpy(test_results.to_numpy().astype(np.float32))

        img = tio.ScalarImage(self.image_path + str(self.images[idx]))
        
        #get image and caption by id
        labels = self.labels[self.labels["Image Data ID"] == self.images[idx].split(".")[0]].Group
        labels = labels.map(idx_to_label)
        labels = torch.tensor(labels.values.astype(np.float32))
        
        if self.transform is not None:
            img = self.transform(img)

        return img.data, test_results, labels 