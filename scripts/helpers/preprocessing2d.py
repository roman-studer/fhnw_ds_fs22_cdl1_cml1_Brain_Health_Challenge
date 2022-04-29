import torchio.transforms as transforms
import torchvision.transforms as pytransforms
import sys
sys.path.insert(0,'../helpers/')
from helpers import miscellaneous as misc


def get_transformer(transformer_name):
    if transformer_name == 'None':
        return None, None

    elif transformer_name == 'Test':
        return _test_transformer(), _test_transformer()
    elif transformer_name == 'Crop':
        return _crop_transformer(), _crop_transformer()
    elif transformer_name == 'Crop_Augment':
        return _crop_transformer(), _crop_augment_transformer()
    else:
        raise ValueError('Transformer is invalid or non-existent')


def _test_transformer():
    return transforms.Compose([
        transforms.RescaleIntensity((-1, 1)),
        transforms.Resample((1,1,1)),
        transforms.Resize((150,150,1)),
    ])

# for 2d images
def _crop_transformer():
    config = misc.get_config()
    return transforms.Compose(
        [
           # transforms.NormalizationTransform((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            transforms.Resize((1, config['IMAGE_RESIZE'],config['IMAGE_RESIZE'])),
        ]
    )

# for 2d images
def _crop_augment_transformer():
    config = misc.get_config()
    return transforms.Compose(
        [
            transforms.CropOrPad((config['IMAGE_HEIGHT'], config['IMAGE_WIDTH'])),
            transforms.NormalizationTransform((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            pytransforms.AutoAugment(),
            transforms.Resize(config['IMAGE_RESIZE']),
        ]
    )
