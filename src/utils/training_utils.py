import sys
sys.path.append('/home/derek/Desktop/RSNA_baseline_kaggle/src')
import numpy as np
import pandas as pd
import torch
from model.BreastCancerModel import *
from dataset.BreastCancerDataSet import *
from config.config import *
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from utils.model_utils import *
from dataset.augmentation import *
import gc
import wandb
from utils.sampler import * 
from torch.utils.data.sampler import SequentialSampler
    
# def gc_collect():
#     gc.collect()
#     torch.cuda.empty_cache()
    
# def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
#     decay = []
#     no_decay = []
#     for name, param in model.named_parameters():
#         if not param.requires_grad:
#             continue
#         if len(param.shape) == 1 or np.any([v in name.lower()  for v in skip_list]):
#             # print(name, 'no decay')
#             no_decay.append(param)
#         else:
#             # print(name, 'decay')
#             decay.append(param)
#     return [
#         {'params': no_decay, 'weight_decay': 0.},
#         {'params': decay, 'weight_decay': weight_decay}]
    

# def evaluate_model(model: BreastCancerModel, ds, max_batches=PREDICT_MAX_BATCHES, shuffle=False, config=Config):
#     # torch.manual_seed(42)
#     model = model.to(DEVICE)
    
#     dl_test = torch.utils.data.DataLoader(ds, batch_size=config.VAL_BATCH_SIZE, shuffle=shuffle,
#                                           num_workers=NUM_WORKERS, pin_memory=False,
#                                           sampler=SequentialSampler(ds))
#     pred_cancer = []
#     with torch.no_grad():
        
#         model.eval()
#         cancer_losses = []
#         aux_losses = []
#         losses = []
#         targets = []
#         with tqdm(dl_test, desc='Eval', mininterval=30) as progress:
#             for i, (X, y_cancer, y_aux) in enumerate(progress):
#                 with autocast(enabled=True):
#                     y_aux = y_aux.to(DEVICE)
#                     X = X.to(DEVICE)
#                     y_cancer_pred, aux_pred = model.forward(X)
#                     if config.TTA:
#                         y_cancer_pred2, aux_pred2 = model.forward(torch.flip(X, dims=[-1])) # horizontal mirror
#                         y_cancer_pred = (y_cancer_pred + y_cancer_pred2) / 2
#                         aux_pred = [(v1 + v2) / 2 for v1, v2 in zip(aux_pred, aux_pred2)]

#                     cancer_loss = torch.nn.functional.binary_cross_entropy_with_logits(
#                         y_cancer_pred, 
#                         y_cancer.to(float).to(DEVICE),
#                         pos_weight=torch.tensor([config.POSITIVE_TARGET_WEIGHT]).to(DEVICE)
#                     ).item()
#                     aux_loss = torch.mean(torch.stack([torch.nn.functional.cross_entropy(aux_pred[i], y_aux[:, i]) for i in range(y_aux.shape[-1])])).item()
#                     pred_cancer.append(torch.sigmoid(y_cancer_pred))
#                     cancer_losses.append(cancer_loss)
#                     aux_losses.append(aux_loss)
#                     losses.append(cancer_loss + config.AUX_LOSS_WEIGHT * aux_loss)
#                     targets.append(y_cancer.cpu().numpy())
#                 if i >= max_batches:
#                     break
#         targets = np.concatenate(targets)
#         pred = torch.concat(pred_cancer).cpu().numpy()
#         pf1, thres = optimal_f1(targets, pred)
#         return np.mean(cancer_losses), (pf1, thres), pred, np.mean(losses), np.mean(aux_losses)
    
# def train_model(ds_train, ds_eval, logger, name, config=Config, do_save_model=True):
#     # torch.manual_seed(42)
#     ewrs = ExhaustiveWeightedRandomSampler(weights=ds_train.weight,
#                                         num_samples=len(ds_train.weight),
#                                         exaustive_weight=1,
#                                         generator=None)
#     dl_train = torch.utils.data.DataLoader(ds_train, batch_size=config.TRAIN_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,\
#                                            pin_memory=False, sampler=ewrs, drop_last=True)

#     model = BreastCancerModel(AUX_TARGET_NCLASSES, config.MODEL_TYPE, config.DROPOUT).to(DEVICE)

#     if config.ADAMW:
#         optim = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
#     else:
#         optim = torch.optim.Adam(model.parameters())


#     scheduler = None
#     if config.ONE_CYCLE:
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=config.ONE_CYCLE_MAX_LR, epochs=config.EPOCHS,
#                                                         steps_per_epoch=int(len(ds_train)/config.TRAIN_BATCH_SIZE),
#                                                         pct_start=config.ONE_CYCLE_PCT_START, 
#                                                         anneal_strategy=config.ANNEAL_STRATEGY,
#                                                         div_factor=config.LR_DIV,
#                                                         final_div_factor=config.LR_FINAL_DIV)
                                                        
    

#     scaler = GradScaler()
#     best_eval_score = 0
#     optim.zero_grad()
    
    
#     for epoch in tqdm(range(config.EPOCHS), desc='Epoch'):
#         model.train()
#         with tqdm(dl_train, desc='Train', mininterval=30) as train_progress:
#             for batch_idx, (X, y_cancer, y_aux) in enumerate(train_progress):
#                 y_aux = y_aux.to(DEVICE)

#                 # optim.zero_grad()
#                 torch.set_grad_enabled(True)
#                 # Using mixed precision training
#                 with autocast():
#                     y_cancer_pred, aux_pred = model.forward(X.to(DEVICE))
#                     cancer_loss = torch.nn.functional.binary_cross_entropy_with_logits(
#                         y_cancer_pred,
#                         y_cancer.to(float).to(DEVICE),
#                         pos_weight=torch.tensor([config.POSITIVE_TARGET_WEIGHT]).to(DEVICE)
#                     )
#                     aux_loss = torch.mean(torch.stack([torch.nn.functional.cross_entropy(aux_pred[i], y_aux[:, i]) for i in range(y_aux.shape[-1])]))
#                     loss = cancer_loss + config.AUX_LOSS_WEIGHT * aux_loss
#                     if np.isinf(loss.item()) or np.isnan(loss.item()):
#                         print(f'Bad loss, skipping the batch {batch_idx}')
#                         del loss, cancer_loss, y_cancer_pred
#                         gc_collect()
#                         continue

#                 # scaler is needed to prevent "gradient underflow"
#                 # scaler.scale(loss).backward()
#                 # scaler.step(optim)
#                 # if scheduler is not None:
#                 #     scheduler.step()
                    
#                 # scaler.update()
#                 scaler.scale(loss).backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
#                 scaler.step(optim)
#                 scaler.update()
#                 optim.zero_grad()
#                 scheduler.step()
                

#                 lr = scheduler.get_last_lr()[0] if scheduler else config.ONE_CYCLE_MAX_LR
#                 logger.log({'loss': (loss.item()),
#                             'cancer_loss': cancer_loss.item(),
#                             'aux_loss': aux_loss.item(),
#                             'lr': lr,
#                             'epoch': epoch})


#         if ds_eval is not None and MAX_EVAL_BATCHES > 0:
#             cancer_loss, (f1, thres), _, loss, aux_loss = evaluate_model(
#                 model, ds_eval, max_batches=MAX_EVAL_BATCHES, shuffle=False, config=config)

#             if f1 > best_eval_score:
#                 best_eval_score = f1
#                 if do_save_model:
#                     save_model(name, model, thres, config.MODEL_TYPE)
#                     art = wandb.Artifact("rsna-breast-cancer", type="model")
#                     art.add_file(f'{name}')
#                     logger.log_artifact(art)

#             logger.log(
#                 {
#                     'eval_cancer_loss': cancer_loss,
#                     'eval_f1': f1,
#                     'max_eval_f1': best_eval_score,
#                     'eval_f1_thres': thres,
#                     'eval_loss': loss,
#                     'eval_aux_loss': aux_loss,
#                     'epoch': epoch
#                 }
#             )

#     return model


    
# if __name__ == "__main__":
#     df_train = pd.read_csv('src/5folds_train.csv')
#     ds_train = BreastCancerDataSet(df_train, TRAIN_IMAGES_PATH, True)
#     AUX_TARGET_NCLASSES = df_train[CATEGORY_AUX_TARGETS].max() + 1
    
#     m = BreastCancerModel(AUX_TARGET_NCLASSES)
#     closs, f1, pred, loss, aloss = evaluate_model(m, ds_train, max_batches=2)
#     del m
#     closs, f1, pred.shape, loss, aloss