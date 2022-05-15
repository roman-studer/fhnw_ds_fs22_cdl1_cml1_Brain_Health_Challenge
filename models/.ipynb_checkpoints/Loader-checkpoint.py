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
    def __init__(self, dataset_path, transform=None):
        """
        Parameters:
            image_path (string): Path to the image folder
            test_path (string: Path to where tabular data is stored
            annotation_path (string): path to annotation where label is stored
            transform (object, callable): Optional transform to be applied
                on a sample.
        """
        self.dataset_path = dataset_path
        self.transform = transform
        # fixme should be handled in another way. maybe in configuration file?
        self.idx_to_label = {
            'CN': 0,
            'MCI': 1,
            'AD': 2
        }
        
        
        self.df = pd.read_csv(self.dataset_path)
        
        # fix this while generating train/test split
        self.df.dropna(subset=['filename'], how='all', inplace=True)

    def __len__(self):
        """
        returns amount of images in total
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Parameters:
            idx (int): index to image
        """
        
        #switch case for each individual cognitive test?
        """test_results = self.test_results[["CDMEMORY", "CDORIENT", "CDJUDGE", "CDCOMMUN", "CDHOME", "CDCARE", "CDGLOBAL"]].sample(n=1)
        test_results = torch.from_numpy(test_results.to_numpy().astype(np.float32))"""

        img = tio.ScalarImage(self.df.iloc[idx].filename)
        
        #get image and caption by id
        label = self.df.iloc[idx].Group
        label = self.idx_to_label[label]
        label = torch.tensor(label).to(torch.float32)
        
        if self.transform is not None:
            img = self.transform(img)

        return {"images": img.data, "labels": label} #, test_results