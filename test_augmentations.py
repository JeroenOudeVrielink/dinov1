import utils
from torchvision import transforms
from PIL import Image
from torchvision.transforms import functional as F
from main_dino import EdgePreservingFilter

import cv2

IMG_PATH = "test_imgs/3.png"


flip_and_color_jitter = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                )
            ],
            p=1,
        ),
        transforms.RandomGrayscale(p=0.2),
    ]
)

global_crop_aug_1 = [
    transforms.RandomResizedCrop(
        224,
        scale=(0.14, 1),
        interpolation=Image.BICUBIC,
    ),
    flip_and_color_jitter,
    # utils.GaussianBlur(1.0),
    EdgePreservingFilter(),
]
global_crop1 = transforms.Compose(global_crop_aug_1)


global_crop_aug_2 = [
    transforms.RandomResizedCrop(
        224,
        scale=(0.14, 1),
        interpolation=Image.BICUBIC,
    ),
    flip_and_color_jitter,
    utils.GaussianBlur(0.0),
    utils.Solarization(1.0),
]
global_crop2 = transforms.Compose(global_crop_aug_2)


local_crop_aug = [
    transforms.RandomResizedCrop(
        96,
        scale=(0.05, 0.14),
        interpolation=Image.BICUBIC,
    ),
    flip_and_color_jitter,
    # utils.GaussianBlur(p=0.5),
    EdgePreservingFilter(sigma_s=60, sigma_r=0.4),
]
local_crop = transforms.Compose(local_crop_aug)

img = Image.open(IMG_PATH)

img.show(title="original")
# cv2.waitKey(0)

global_crop1_img = global_crop1(img)
global_crop1_img.show(title="flip, collor jitter, gaussian blur")

# global_crop2_img = global_crop2(img)
# global_crop2_img.show(title="flip, collor jitter, gaussian blur, solarization")

# local_crop_img = local_crop(img)
# local_crop_img.show(title="flip, collor jitter, gaussian blur")


# img_color_jitter = flip_and_color_jitter(img)
# img_color_jitter.show(title="flip, collor jitter")
