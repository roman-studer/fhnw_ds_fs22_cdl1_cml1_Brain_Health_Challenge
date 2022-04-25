import torchvision.transforms as transforms
import sys
sys.path.insert(0,'../helpers/')
from helpers import miscellaneous as misc


def get_transformer(transformer_name):
    if transformer_name == 'None':
        return None

    elif transformer_name == 'Test':
        return _test_transformer(), _test_transformer()
    elif transformer_name == 'Crop':
        return _crop_transformer(), _crop_transformer()
    elif transformer_name == 'Crop_Augment':
        return _crop_transformer(), _crop_augment_transformer()

    else:
        raise ValueError('Transformer is invalid or non-existent')


def _test_transformer():
    return transforms.Compose(
        [transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])


def _crop_transformer():
    config = misc.get_config()
    return transforms.Compose(
        [
            transforms.CenterCrop((config['IMAGE_HEIGHT'], config['IMAGE_WIDTH'])),
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            transforms.Resize(config['IMAGE_RESIZE']),
        ]
    )


def _crop_augment_transformer():
    config = misc.get_config()
    return transforms.Compose(
        [
            transforms.CenterCrop((config['IMAGE_HEIGHT'], config['IMAGE_WIDTH'])),
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            transforms.AutoAugment(),
            transforms.Resize(config['IMAGE_RESIZE']),
        ]
    )
