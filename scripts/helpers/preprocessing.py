#import Augmentor
import yaml
import numpy as np

class ADNIPreprocessing():

    def __init__(self, config, device='gpu'):
        self._device = device
        self._config = config

    def load_train_test_set(self):
        raise NotImplementedError

    def load_dataset(self):
        raise NotImplementedError

    def get_by_ID(self, patient_ID):
        raise NotImplementedError

    def get_by_range(self, range):
        raise NotImplementedError
   
    def image_augmentation(self, path_folder: str, n_samples: int):
        """
        Taken (and adapted) from Susanne Suter.
        Returns n_samples of augmented images based on the reference images in path_folder
        
        :param path_folder: str, reference images
        :param n_samples: int, number of samples to create
        :return: n_samples of images
        """
        p = Augmentor.Pipeline(path_folder)
        p.rotate(probability=0.3, max_left_rotation=4, max_right_rotation=4)
        p.zoom(probability=0.3, min_factor=0.7, max_factor=1.2)
        p.random_distortion(probability=0.3, grid_width=4, grid_height=4, magnitude=6)
        p.random_brightness(probability=0.8, min_factor=0.5, max_factor=2)
        p.random_color(probability=0.3, min_factor=0.5, max_factor=1.5)
        p.random_contrast(probability=0.3, min_factor=0.8, max_factor=1.2)

        p.sample(n_samples, multi_threaded=False)

    def mean_center_images(self):
        # center image with "mean-image" or "per-channel-mean"
        raise NotImplementedError
        
    def normalise(image):
        # normalise and clip images -1000 to 800
        np_img = np.clip(image, -1000., 800.).astype(np.float32)
        return np_img