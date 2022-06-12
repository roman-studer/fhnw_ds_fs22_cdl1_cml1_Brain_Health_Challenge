import torchio.transforms as transforms
import torchvision.transforms as pytransforms
import sys
sys.path.insert(0,'../helpers/')
from helpers import miscellaneous as misc
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    EnsureType,
)


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
    config = misc.get_config()
    return transforms.Compose([
        transforms.RescaleIntensity((-1, 1)),
        transforms.Resample((1,1,1)),
        transforms.Resize((config['IMAGE_RESIZE1'],config['IMAGE_RESIZE2'], 1)),
    ])

# for 2d images
def _crop_transformer():
    config = misc.get_config()
    return transforms.Compose(
        [
            ScaleIntensity(),
            EnsureType(),
            pytransforms.Resize((config['IMAGE_RESIZE1'],config['IMAGE_RESIZE2']))
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
            transforms.Resize((config['IMAGE_RESIZE1'],config['IMAGE_RESIZE2'])),
        ]
    )
