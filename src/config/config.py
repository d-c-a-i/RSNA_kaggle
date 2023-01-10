
import os
import numpy as np
import torch 
import pandas as pd
import yaml
import argparse

parser = argparse.ArgumentParser(description='RSNA Breast Cancer Detection')
parser.add_argument("--cfg",\
    default='/home/derek/Desktop/RSNA_breast_cancer/RSNA/configs/yaml_configs/tf_efficientnetv2_s_512_1.yaml')

args = parser.parse_args()

with open(args.cfg) as f:
    CFG = yaml.load(f, Loader=yaml.FullLoader)

class Config:
    # These are optimal parameters collected from https://wandb.ai/vslaykovsky/rsna-breast-cancer-sweeps/sweeps/k281hlr9?workspace=user-vslaykovsky
    ONE_CYCLE = True
    ONE_CYCLE_PCT_START = 0.1
    ADAMW = True
    # ADAMW_DECAY = 0.024
    ONE_CYCLE_MAX_LR = 3e-4
    EPOCHS = 10
    ANNEAL_STRATEGY = 'cos'
    LR = 3e-4
    LR_DIV = 1.0
    LR_FINAL_DIV = 10000.0
    WEIGHT_DECAY = 1e-2
    MODEL_TYPE = 'tf_efficientnetv2_s'
    DROPOUT = 0.2
    DROP_PATH_RATE = 0.0
    AUG = True
    AUX_LOSS_WEIGHT = 5
    POSITIVE_TARGET_WEIGHT=1
    # BATCH_SIZE = 32
    TRAIN_BATCH_SIZE = 16
    VAL_BATCH_SIZE = 16
    AUTO_AUG_M = 10
    AUTO_AUG_N = 2
    TTA = False
    
RSNA_2022_PATH = '/home/derek/ML_comp_data/RSNA'
TRAIN_IMAGES_PATH = f'/home/derek/Desktop/rsna-cut-off-empty-space-from-images'
MAX_TRAIN_BATCHES = 40000
MAX_EVAL_BATCHES = 400
MODELS_PATH = '/home/derek/Desktop/RSNA_baseline_kaggle/models_roi_1024'
NUM_WORKERS = 8
PREDICT_MAX_BATCHES = 1e9
N_FOLDS = 5
FOLDS = np.array(os.environ.get('FOLDS', '0,1,2,3,4').split(',')).astype(int)
WANDB_SWEEP_PROJECT = 'rsna-breast-cancer-sweeps'
WANDB_PROJECT = 'RSNA-breast-cancer-v4'

CATEGORY_AUX_TARGETS = ['view', 'implant', 'age']
TARGET = 'cancer'
ALL_FEAT = [TARGET] + CATEGORY_AUX_TARGETS


DEBUG = os.environ.get('DEBUG', 'true').lower() == 'true'
WANDB_SWEEP = False
TRAIN = False
CV = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

df_train = pd.read_csv('/home/derek/ML_comp_data/RSNA/split/baseline/5folds_train.csv')
AUX_TARGET_NCLASSES = df_train[CATEGORY_AUX_TARGETS].max() + 1



BAD_IMAGES = [
            '50203_643148078.png',
            '38739_1189630231.png',
            '65471_1729119684.png',
            '33581_1586149541.png',
            '33581_357843412.png',
            '25323_1743461841.png',
            '3768_1634189725.png',
            '1511_764545189.png',
            '38739_1110010839.png',
            '27306_2057264782.png',
            '1511_1031853445.png',
            '16497_1541367572.png',
            '17111_543347978.png',
            '31401_884457810.png',
            '33084_1990776518.png',
            '822_1942326353.png', 
            ]