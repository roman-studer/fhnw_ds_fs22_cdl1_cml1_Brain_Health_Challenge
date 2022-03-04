import yaml
import os


def get_config() -> dict:
    """
    Returns a dictionary of the config.yml file in root folder
    :return: dictionary of config.yml
    """
    with open('../../CONFIG.yml', 'r') as f:
        c = yaml.safe_load(f)
    return c


def flatten_data(raw_data_dir=None):
    if raw_data_dir is None:
        try:
            raw_data_dir = get_config()["RAW_DATA_DIR"]
        except:
            raise ValueError(
                'Could not find CONFIG.yml. Provide a path to the file with the "raw_data_path" parameter')

    raise NotImplementedError

def get_file_names(path: str, suffix: str = None):
    """
    path: str, path to search from
    suffix: str, filters for files with provided suffix (e.g. ".nii")
    :return: A list of files with suffix "suffix" in provided path (including substructure)
    """
    filenames = []
    for _, _, files in os.walk(path):
        if suffix:
            for name in files:
                if name.endswith(suffix):
                    filenames.append(name)

    return filenames
