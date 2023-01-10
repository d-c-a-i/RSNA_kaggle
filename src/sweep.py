import os
import sys
sys.path.append('/home/derek/Desktop/RSNA_baseline_kaggle/src')
import wandb
from config.config import *
from dataset.BreastCancerDataSet import *
from utils.training_utils import *

if __name__ == "__main__":
   sweep_id = os.environ.get('SWEEP_ID')
   print('wandb sweep ', sweep_id)
   
   if sweep_id is None:
        """
        First run. Generate sweep_id.
        """
        sweep_id = wandb.sweep(sweep={
            'method': 'bayes',
            'name': 'rsna-sweep',
            'metric': {'goal': 'maximize', 'name': 'max_eval_f1'},
            'parameters':
                {
                    'ONE_CYCLE': {'values': [True, False]},
                    'ONE_CYCLE_PCT_START': {'values': [0.1]},
                    'ADAMW': {'values': [True, False]},
                    'ADAMW_DECAY': {'min': 0.001, 'max': 0.1, 'distribution': 'log_uniform_values'},
                    'ONE_CYCLE_MAX_LR': {'min': 1e-5, 'max': 1e-3, 'distribution': 'log_uniform_values'},
                    'EPOCHS': {'min': 1, 'max': 12, 'distribution': 'q_log_uniform_values'},
                    'MODEL_TYPE': {'values': ['resnext50_32x4d', 'efficientnetv2_rw_s', 'seresnext50_32x4d', 'inception_v4', 'efficientnet_b4']},
                    'DROPOUT': {'values': [0., 0.2]},
                    'AUG': {'values': [True, False]},
                    'AUX_LOSS_WEIGHT': {'min': 0.01, 'max': 100., 'distribution': 'log_uniform_values'},
                    'POSITIVE_TARGET_WEIGHT': {'min': 1., 'max': 60., 'distribution': 'uniform'},
                    'BATCH_SIZE': {'values': [32]},
                    'AUTO_AUG_M': {'min': 1, 'max': 20, 'distribution': 'q_log_uniform_values'},
                    'AUTO_AUG_N': {'min': 1, 'max': 6, 'distribution': 'q_uniform'},
                    'TTA': {'values': [False]},
                }
        }, project=WANDB_SWEEP_PROJECT)
        print('Generated sweep id', sweep_id)
    
    else:
        """
        Agent run. Use sweep_id generated above to produce (semi)-random hyperparameters run.config
        """
        def wandb_callback():
            with wandb.init() as run:
                print('params', run.config)
                fold = 0
                ds_train = BreastCancerDataSet(df_train.query('split != @fold'), TRAIN_IMAGES_PATH, run.config.AUG)
                ds_eval = BreastCancerDataSet(df_train.query('split == @fold'), TRAIN_IMAGES_PATH, False)
                train_model(ds_train, ds_eval, run, f'model-f{fold}', config=run.config, do_save_model=False)


        # Start sweep job.
        wandb.agent(sweep_id, project=WANDB_SWEEP_PROJECT, function=wandb_callback, count=100000)