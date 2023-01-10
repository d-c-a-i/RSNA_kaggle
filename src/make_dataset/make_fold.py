import pandas as pd
import sys
sys.path.append('/home/derek/Desktop/RSNA_baseline_kaggle/src')
from config.config import *
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
import os

def split_data(df_train, save_dir, save_folder):
    os.makedirs(os.path.join(save_dir, save_folder), exist_ok=True)
    # TO-DO: FILTER BAD IMAGES
    split = StratifiedGroupKFold(N_FOLDS)
    for k, (_, test_idx) in enumerate(split.split(df_train, df_train.cancer, groups=df_train.patient_id)):
        df_train.loc[test_idx, 'split'] = k
    df_train.split = df_train.split.astype(int)
    print(df_train.groupby('split').cancer.mean())
    df_train.age.fillna(df_train.age.mean(), inplace=True)
    df_train['age'] = pd.qcut(df_train.age, 10, labels=range(10), retbins=False).astype(int)
    df_train[CATEGORY_AUX_TARGETS] = df_train[CATEGORY_AUX_TARGETS].apply(LabelEncoder().fit_transform) 
    df_train["weight"] = 1
    df_train.loc[df_train.cancer == 1, "weight"] = len(df_train.loc[df_train.cancer == 0]) / len(df_train.loc[df_train.cancer == 1])
    #delete AT, LM, ML, LMO views
    df_train = df_train.drop(df_train[df_train.view == 'AT'].index)
    df_train = df_train.drop(df_train[df_train.view == 'LM'].index)
    df_train = df_train.drop(df_train[df_train.view == 'ML'].index)
    df_train = df_train.drop(df_train[df_train.view == 'LMO'].index)
    
    df_train.to_csv(os.path.join(save_dir, save_folder, f'{N_FOLDS}folds_train.csv'))
    
    
    
if __name__ == "__main__":
    # df_train = pd.read_csv((f'{RSNA_2022_PATH}/train.csv'))
    # split_data(df_train, '/home/derek/ML_comp_data/RSNA/split', 'baseline')
    import pandas as pd
    pd.read_csv('src/5folds_train.csv')
    print(len(df_train.loc[df_train.cancer == 0]) / len(df_train.loc[df_train.cancer == 1]))