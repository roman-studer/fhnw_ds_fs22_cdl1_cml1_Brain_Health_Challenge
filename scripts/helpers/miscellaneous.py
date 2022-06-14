import yaml
import os
import nibabel as nib
import shutil
import tqdm
import glob
from pathlib import Path
import torch

def get_config() -> dict:
    """
    Returns a dictionary of the config.yml file in root folder
    :return: dictionary of config.yml
    """
    try:
        with open('../../../CONFIG.yml', 'r') as f:
            c = yaml.safe_load(f)
        return c
    except:
        pass
    try:
        with open('../../CONFIG.yml', 'r') as f:
            c = yaml.safe_load(f)
        return c
    except:
        pass

    try:
        with open('../CONFIG.yml','r') as f:
            c = yaml.safe_load(f)
        return c
    except:
        pass

    try:
        with open('CONFIG.yml','r') as f:
            c = yaml.safe_load(f)
        return c
    except:
        raise FileNotFoundError(
            'Could not find CONFIG.yml file. Try importing it manually')


def flatten_data() -> None:
    """
    Copies all .nii files in CONFIG["RAW_DATA_DIR"] to CONFIG["FLATTENED_FOLDER"]
    :return: None
    """
    try:
        config = get_config()
    except:
        raise ValueError(
            'Could not find CONFIG.yml. Provide a path to the file with the "raw_data_path" parameter')

    raw_data_dir = config["RAW_DATA_DIR"]
    flattened_data_dir = config["FLATTENED_FOLDER"]

    nii_files = glob.glob('../../' + raw_data_dir + '**/*.nii', recursive=True)
    for file in tqdm(nii_files):
        shutil.copy2(Path(file), flattened_data_dir)


def get_file_names(path: str, suffix: str = None, return_all: bool = False):
    """
    path: str, path to search from
    suffix: str, filters for files with provided suffix (e.g. ".nii")
    :return: A list of files with suffix "suffix" in provided path (including substructure)
    """
    filenames = []
    for root, dirs, files in os.walk(path):
        if suffix:
            for name in files:
                if name.endswith(suffix):
                    filenames.append(name)

    if return_all:
        return (root, dirs, filenames)

    return filenames

def get_nii_filenames(path: str):
    return glob.glob('../../' + path + '**/*.nii', recursive=True)

def get_png_filenames(path: str):
    return glob.glob('../../' + path + '**/*.png', recursive=True)

def get_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header

    Parameters
    ----------

    img_path: string
    String with the path of the 'nii' or 'nii.gz' image file name.

    Returns
    -------
    Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header

def save_model(model, PATH):
    torch.save(model.state_dict(), PATH)


def load_model(model, PATH, NAME):
    model.load_state_dict(torch.load(PATH+NAME))
    model.eval()
    return model