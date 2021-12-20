import albumentations as A
from albumentations.pytorch import ToTensorV2

class CIFAR10Transforms:
    def __init__(self):
        pass

    def build_transforms(self,  train_list=[], test_list=[]):
        train_list.extend([A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ToTensorV2()])
        test_list.extend([A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ToTensorV2()])
        return A.Compose(train_tfms_list), A.Compose(test_tfms_list)
