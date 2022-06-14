import torchio.transforms as transforms
import torchvision.transforms as pytransforms
import sys
import numpy as np
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
    Rand2DElastic,
    RandZoom,
    ScaleIntensity,
    EnsureType,
    NormalizeIntensity,
    RandGaussianSmooth,
    Resize
)


def get_transformer(transformer_name):
    if transformer_name == 'None':
        return None, None

    elif transformer_name == 'Test':
        return _test_transformer(), _test_transformer()
    elif transformer_name == 'Crop':
        return _crop_transformer(), _crop_transformer()
    elif transformer_name == 'Crop_Augment':
        return _crop_augment_transformer(),  _crop_transformer()
    elif transformer_name == 'Monai_Augment':
        return _monai_augment_transformer(),  _monai_transformer_test()
    elif transformer_name == 'Monai_Blur':
        return _monai_augment_blur_transformer(), _monai_transformer_test()
    else:
        raise ValueError('Transformer is invalid or non-existent')


def _test_transformer():
    config = misc.get_config()
    return transforms.Compose([
        transforms.RescaleIntensity((-1, 1)),
        transforms.Resample((1,1,1)),
        Resize((config['IMAGE_RESIZE1'],config['IMAGE_RESIZE2'], 1)),
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
            pytransforms.Resize((config['IMAGE_RESIZE1'],config['IMAGE_RESIZE2']))
        ]
    )


def _monai_augment_transformer():
    config = misc.get_config()
    return transforms.Compose(
        [
            NormalizeIntensity(),
            #ScaleIntensity(-1, 1, -1024, 1024),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.3),
            #Rand2DElastic(prob=0.3, spacing=(40,40), magnitude_range=(0,1), padding_mode='zeros'),
            EnsureType(),
            #pytransforms.Resize((config['IMAGE_RESIZE1'],config['IMAGE_RESIZE2']))
        ]
    )

def _monai_transformer_test():
    config = misc.get_config()
    return transforms.Compose(
        [
            #NormalizeIntensity(),
            #ScaleIntensity(-1, 1, -1024, 1024),
            EnsureType(),
            Resize((config['IMAGE_RESIZE1'],config['IMAGE_RESIZE2']))
        ]
    )


def _monai_augment_blur_transformer():    
    config = misc.get_config()
    return transforms.Compose(
        [
            #NormalizeIntensity(),
            #ScaleIntensity(-1, 1, -1024, 1024),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            #Rand2DElastic(prob=0.3, spacing=(40,40), magnitude_range=(0,1), padding_mode='zeros'),
            RandGaussianSmooth(prob=0.5),
            RandRotate(prob=0.5),
            RandFlip(prob=0.4),
            EnsureType(),
            Resize((config['IMAGE_RESIZE1'],config['IMAGE_RESIZE2']))
        ]
    )