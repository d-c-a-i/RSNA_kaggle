import torch
from timm import create_model
import pandas as pd
import sys
sys.path.append('/home/derek/Desktop/RSNA_baseline_kaggle/src')
from config.config import *

class BreastCancerModel(torch.nn.Module):
    def __init__(self, aux_classes, model_type=Config.MODEL_TYPE, dropout=0., drop_path_rate=0.):
        super().__init__()
        self.model = create_model(model_type, pretrained=True, num_classes=0, drop_rate=dropout, drop_path_rate = drop_path_rate)

        self.backbone_dim = self.model(torch.randn(1, 3, 512, 512)).shape[-1]

        self.nn_cancer = torch.nn.Sequential(
            torch.nn.Linear(self.backbone_dim, 1),
        )
        self.nn_aux = torch.nn.ModuleList([
            torch.nn.Linear(self.backbone_dim, n) for n in aux_classes
        ])

    def forward(self, x):
        # returns logits
        x = self.model(x)

        cancer = self.nn_cancer(x).squeeze()
        aux = []
        for nn in self.nn_aux:
            aux.append(nn(x).squeeze())
        return cancer, aux

    def predict(self, x):
        cancer, aux = self.forward(x)
        sigaux = []
        for a in aux:
            sigaux.append(torch.softmax(a, dim=-1))
        return torch.sigmoid(cancer), sigaux


if __name__ == '__main__':
    df_train = pd.read_csv('src/5folds_train.csv')
    AUX_TARGET_NCLASSES = df_train[CATEGORY_AUX_TARGETS].max() + 1
    with torch.no_grad():
        model = BreastCancerModel(AUX_TARGET_NCLASSES, model_type='seresnext50_32x4d')
        pred, aux = model.predict(torch.randn(2, 3, 512, 512))
        print('seresnext', pred.shape, len(aux))

        model = BreastCancerModel(AUX_TARGET_NCLASSES, model_type='efficientnet_b4')
        pred, aux = model.predict(torch.randn(2, 3, 512, 512))
        print('efficientnet_b4', pred.shape, len(aux))

    del model