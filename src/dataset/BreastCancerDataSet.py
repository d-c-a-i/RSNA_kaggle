import sys
sys.path.append('/home/derek/Desktop/RSNA_baseline_kaggle/src')
from config.config import *
from PIL import Image
import torch
import pandas as pd
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import cv2
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class BreastCancerDataSet(torch.utils.data.Dataset):
    def __init__(self, df, path, transform=False):
        super().__init__()
        self.df = df
        self.path = path
        self.weight = df.weight.tolist()
        if transform == True:
            self.transforms = albu.Compose([
            albu.RandomResizedCrop(height=1024, width=512, scale=(0.8, 1.0), ratio=(0.45, 0.55), interpolation=cv2.INTER_NEAREST, p=1.0),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.CoarseDropout(max_holes=10, max_height=32, max_width=32, min_height=16, min_width=16, fill_value=0, p=0.5),  #change based on image
            albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ])
        else:
            self.transforms = albu.Compose([
            albu.Resize(1024, 512),
            albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ToTensorV2(),
        ])

    def __getitem__(self, i):

        path = f'{self.path}/{self.df.iloc[i].patient_id}/{self.df.iloc[i].image_id}.png'
        try:
            # img = Image.open(path).convert('RGB')
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = np.stack([img, img, img], axis=-1)
            img = self.transforms(image=img)['image']
        except Exception as ex:
            print(path, ex)
            return None


        if TARGET in self.df.columns:
            cancer_target = torch.as_tensor(self.df.iloc[i].cancer)
            cat_aux_targets = torch.as_tensor(self.df.iloc[i][CATEGORY_AUX_TARGETS])
            return img, cancer_target, cat_aux_targets

        return img

    def __len__(self):
        return self.df.shape[0]

if __name__ == "__main__":
    df_train = pd.read_csv('src/5folds_train.csv')
    ds_train = BreastCancerDataSet(df_train, TRAIN_IMAGES_PATH, True)
    if DEBUG:
        X, y_cancer, y_aux = ds_train[42]
        print(X.shape, y_cancer.shape, y_aux.shape)

