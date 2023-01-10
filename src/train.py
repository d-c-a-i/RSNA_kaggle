import sys
sys.path.append('/home/derek/Desktop/RSNA_baseline_kaggle/src')
import wandb
from config.config import *
from model.BreastCancerModel import *
from dataset.BreastCancerDataSet import *
from utils.training_utils import *


WANDB_RUN_NAME = f'{Config.MODEL_TYPE}_lr{Config.ONE_CYCLE_MAX_LR}_ep{Config.EPOCHS}_bs{Config.TRAIN_BATCH_SIZE}_pw{Config.POSITIVE_TARGET_WEIGHT}_' +\
f'aux{Config.AUX_LOSS_WEIGHT}_{"adamw" if Config.ADAMW else "adam"}_{"aug" if Config.AUG else "noaug"}_drop{Config.DROPOUT}'
print('run', WANDB_RUN_NAME, 'folds', FOLDS)

git init
if __name__ == "__main__":
    set_seeds(2023)
    df_train = pd.read_csv('/home/derek/ML_comp_data/RSNA/split/baseline/5folds_train.csv')
    AUX_TARGET_NCLASSES = df_train[CATEGORY_AUX_TARGETS].max() + 1
    weight_path = '/home/derek/Desktop/RSNA_baseline_kaggle/exp0_weights'
    os.makedirs(weight_path, exist_ok=True)
    for fold in FOLDS:
        name = f'{WANDB_RUN_NAME}-f{fold}'
        with wandb.init(project=WANDB_PROJECT, name=name, group=WANDB_RUN_NAME) as run:
            gc_collect()
            ds_train = BreastCancerDataSet(df_train.query('split != @fold'), TRAIN_IMAGES_PATH, Config.AUG)
            ds_eval = BreastCancerDataSet(df_train.query('split == @fold'), TRAIN_IMAGES_PATH, False)
            train_model(ds_train, ds_eval, run, f'{weight_path}/model-f{fold}')
    