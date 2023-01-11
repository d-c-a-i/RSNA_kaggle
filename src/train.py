import sys
sys.path.append('/home/derek/Desktop/RSNA_baseline_kaggle/src')
import wandb
from config.config import *
from model.BreastCancerModel import *
from model.CustomNet import *
from dataset.BreastCancerDataSet import *
from utils.training_utils import *
import argparse
import os



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='RSNA Breast Cancer Detection')

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # parser.add_argument('--epochs', type=int, default=10)
    # parser.add_argument('--train_batch_size', type=int, default=64)
    # parser.add_argument('--val_batch_size', type=int, default=128)

    # # Data, model, and output directories
    # parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    # parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    # parser.add_argument('--train_data_path', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=16)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default='dat')
    parser.add_argument('--model_dir', type=str, default='src/exp0')
    parser.add_argument('--train_data_path', type=str, default='/home/derek/Desktop/rsna-cut-off-empty-space-from-images')
    
    args = parser.parse_args()
    
    WANDB_RUN_NAME = f'{Config.MODEL_TYPE}_lr{Config.ONE_CYCLE_MAX_LR}_ep{args.epochs}_bs{args.train_batch_size}_pw{Config.POSITIVE_TARGET_WEIGHT}_' +\
    f'aux{Config.AUX_LOSS_WEIGHT}_{"adamw" if Config.ADAMW else "adam"}_{"aug" if Config.AUG else "noaug"}_drop{Config.DROPOUT}'
    print('run', WANDB_RUN_NAME, 'folds', FOLDS)
    
    # functions
    def gc_collect():
        gc.collect()
        torch.cuda.empty_cache()
    
    def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if len(param.shape) == 1 or np.any([v in name.lower()  for v in skip_list]):
                # print(name, 'no decay')
                no_decay.append(param)
            else:
                # print(name, 'decay')
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]
        

    def evaluate_model(model, ds, max_batches=PREDICT_MAX_BATCHES, shuffle=False, config=Config):
        # torch.manual_seed(42)
        model = model.to(DEVICE)
        
        dl_test = torch.utils.data.DataLoader(ds, batch_size=args.val_batch_size, shuffle=shuffle,
                                            num_workers=NUM_WORKERS, pin_memory=False,
                                            sampler=SequentialSampler(ds))
        pred_cancer = []
        with torch.no_grad():
            
            model.eval()
            cancer_losses = []
            aux_losses = []
            losses = []
            targets = []
            with tqdm(dl_test, desc='Eval', mininterval=30) as progress:
                for i, (X, y_cancer, y_aux) in enumerate(progress):
                    with autocast(enabled=True):
                        y_aux = y_aux.to(DEVICE)
                        X = X.to(DEVICE)
                        if not config.CUSTOM_NET:
                            y_cancer_pred, aux_pred = model.forward(X)
                            # if config.TTA:
                            #     y_cancer_pred2, aux_pred2 = model.forward(torch.flip(X, dims=[-1])) # horizontal mirror
                            #     y_cancer_pred = (y_cancer_pred + y_cancer_pred2) / 2
                            #     aux_pred = [(v1 + v2) / 2 for v1, v2 in zip(aux_pred, aux_pred2)]

                            cancer_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                y_cancer_pred, 
                                y_cancer.to(float).to(DEVICE),
                                pos_weight=torch.tensor([config.POSITIVE_TARGET_WEIGHT]).to(DEVICE)
                            ).item()
                            aux_loss = torch.mean(torch.stack([torch.nn.functional.cross_entropy(aux_pred[i], y_aux[:, i]) for i in range(y_aux.shape[-1])])).item()
                            total_loss = cancer_loss + config.AUX_LOSS_WEIGHT * aux_loss
                            
                        else:
                            # logits_clf, logits_deeps, logits_hypercol = model.forward(X.to(DEVICE))
                            # cancer_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            #     logits_clf,
                            #     y_cancer.to(float).to(DEVICE),
                            #     pos_weight=torch.tensor([config.POSITIVE_TARGET_WEIGHT]).to(DEVICE)
                            # )
                            # # add hypercol loss after experimenting with only deep supervision
                            # aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            #     logits_deeps,
                            #     y_cancer.to(float).to(DEVICE)
                            # )
                            # total_loss = cancer_loss + aux_loss
                            y_cancer_pred = model.forward(X.to(DEVICE))
                            cancer_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                y_cancer_pred,
                                y_cancer.to(float).to(DEVICE),
                                pos_weight=torch.tensor([config.POSITIVE_TARGET_WEIGHT]).to(DEVICE)
                            )
                            # add hypercol loss after experimenting with only deep supervision
                            aux_loss = 0
                            total_loss = cancer_loss + aux_loss
                        
                        pred_cancer.append(torch.sigmoid(y_cancer_pred))
                        cancer_losses.append(cancer_loss)
                        aux_losses.append(aux_loss)
                        losses.append(total_loss)
                        targets.append(y_cancer.cpu().numpy())
                    if i >= max_batches:
                        break
            targets = np.concatenate(targets)
            pred = torch.concat(pred_cancer).cpu().numpy()
            pf1, thres = optimal_f1(targets, pred)
            return np.mean(cancer_losses), (pf1, thres), pred, np.mean(losses), np.mean(aux_losses)
        
    def train_model(ds_train, ds_eval, logger, name, config=Config, do_save_model=True):
        # torch.manual_seed(42)
        ewrs = ExhaustiveWeightedRandomSampler(weights=ds_train.weight,
                                            num_samples=len(ds_train.weight),
                                            exaustive_weight=1,
                                            generator=None)
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=False, num_workers=NUM_WORKERS,\
                                            pin_memory=False, sampler=ewrs, drop_last=True)

        if not config.CUSTOM_NET:
            model = BreastCancerModel(AUX_TARGET_NCLASSES, config.MODEL_TYPE, config.DROPOUT).to(DEVICE)
        else:
            model = Custom_effnetv2_s(last_k_layers=6,
                                      IMAGENET_pretrained=True,
                                      drop_rate=config.DROPOUT,
                                      drop_path_rate=config.DROP_PATH_RATE).to(DEVICE)

        if config.ADAMW:
            optim = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        else:
            optim = torch.optim.Adam(model.parameters())


        scheduler = None
        if config.ONE_CYCLE:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=config.ONE_CYCLE_MAX_LR, epochs=args.epochs,
                                                            steps_per_epoch=int(len(ds_train)/args.train_batch_size),
                                                            pct_start=config.ONE_CYCLE_PCT_START, 
                                                            anneal_strategy=config.ANNEAL_STRATEGY,
                                                            div_factor=config.LR_DIV,
                                                            final_div_factor=config.LR_FINAL_DIV)
                                                            
        

        scaler = GradScaler()
        best_eval_score = 0
        optim.zero_grad()
        
        
        for epoch in tqdm(range(args.epochs), desc='Epoch'):
            model.train()
            with tqdm(dl_train, desc='Train', mininterval=30) as train_progress:
                for batch_idx, (X, y_cancer, y_aux) in enumerate(train_progress):
                    y_aux = y_aux.to(DEVICE)

                    # optim.zero_grad()
                    torch.set_grad_enabled(True)
                    # Using mixed precision training
                    with autocast():
                        if not config.CUSTOM_NET:
                            y_cancer_pred, aux_pred = model.forward(X.to(DEVICE))
                            cancer_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                y_cancer_pred,
                                y_cancer.to(float).to(DEVICE),
                                pos_weight=torch.tensor([config.POSITIVE_TARGET_WEIGHT]).to(DEVICE)
                            )
                            aux_loss = torch.mean(torch.stack([torch.nn.functional.cross_entropy(aux_pred[i], y_aux[:, i]) for i in range(y_aux.shape[-1])]))
                            loss = cancer_loss + config.AUX_LOSS_WEIGHT * aux_loss
                            if np.isinf(loss.item()) or np.isnan(loss.item()):
                                print(f'Bad loss, skipping the batch {batch_idx}')
                                del loss, cancer_loss, y_cancer_pred
                                gc_collect()
                                continue
                        else:
                            # logits_clf, logits_deeps, logits_hypercol = model.forward(X.to(DEVICE))
                            # cancer_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            #     logits_clf,
                            #     y_cancer.to(float).to(DEVICE),
                            #     pos_weight=torch.tensor([config.POSITIVE_TARGET_WEIGHT]).to(DEVICE)
                            # )
                            # # add hypercol loss after experimenting with only deep supervision
                            # aux_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            #     logits_deeps,
                            #     y_cancer.to(float).to(DEVICE)
                            # )
                            # loss = cancer_loss+aux_loss
                            # # loss = cancer_loss + config.AUX_LOSS_WEIGHT * aux_loss
                            # if np.isinf(loss.item()) or np.isnan(loss.item()):
                            #     print(f'Bad loss, skipping the batch {batch_idx}')
                            #     del loss, cancer_loss, y_cancer_pred
                            #     gc_collect()
                            #     continue
                            logits_clf = model.forward(X.to(DEVICE))
                            cancer_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                                logits_clf,
                                y_cancer.to(float).to(DEVICE),
                                pos_weight=torch.tensor([config.POSITIVE_TARGET_WEIGHT]).to(DEVICE)
                            )
                            # add hypercol loss after experimenting with only deep supervision
                            aux_loss = torch.Tensor([0]).to(DEVICE)
                            loss = cancer_loss+aux_loss
                            # loss = cancer_loss + config.AUX_LOSS_WEIGHT * aux_loss
                            if np.isinf(loss.item()) or np.isnan(loss.item()):
                                print(f'Bad loss, skipping the batch {batch_idx}')
                                del loss, cancer_loss, y_cancer_pred
                                gc_collect()
                                continue
                            

                    # scaler is needed to prevent "gradient underflow"
                    # scaler.scale(loss).backward()
                    # scaler.step(optim)
                    # if scheduler is not None:
                    #     scheduler.step()
                        
                    # scaler.update()
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()
                    scheduler.step()
                    

                    lr = scheduler.get_last_lr()[0] if scheduler else config.ONE_CYCLE_MAX_LR
                    logger.log({'loss': (loss.item()),
                                'cancer_loss': cancer_loss.item(),
                                'aux_loss': aux_loss.item(),
                                'lr': lr,
                                'epoch': epoch})


            if ds_eval is not None and MAX_EVAL_BATCHES > 0:
                cancer_loss, (f1, thres), _, loss, aux_loss = evaluate_model(
                    model, ds_eval, max_batches=MAX_EVAL_BATCHES, shuffle=False, config=config)

                if f1 > best_eval_score:
                    best_eval_score = f1
                    if do_save_model:
                        save_model(name, model, thres, config.MODEL_TYPE)
                        art = wandb.Artifact("rsna-breast-cancer", type="model")
                        art.add_file(f'{name}')
                        logger.log_artifact(art)

                logger.log(
                    {
                        'eval_cancer_loss': cancer_loss,
                        'eval_f1': f1,
                        'max_eval_f1': best_eval_score,
                        'eval_f1_thres': thres,
                        'eval_loss': loss,
                        'eval_aux_loss': aux_loss,
                        'epoch': epoch
                    }
                )

        return model
    
    
    
    
    set_seeds(2023)
    df_train = pd.read_csv('/home/derek/Desktop/RSNA_baseline_kaggle/src/5folds_train.csv')
    AUX_TARGET_NCLASSES = df_train[CATEGORY_AUX_TARGETS].max() + 1
    # weight_path = '/home/derek/Desktop/RSNA_baseline_kaggle/exp0_weights'
    # os.makedirs(weight_path, exist_ok=True)
    for fold in FOLDS:
        name = f'{WANDB_RUN_NAME}-f{fold}'
        with wandb.init(project=WANDB_PROJECT, name=name, group=WANDB_RUN_NAME) as run:
            gc_collect()
            ds_train = BreastCancerDataSet(df_train.query('split != @fold'), args.train_data_path, Config.AUG)
            ds_eval = BreastCancerDataSet(df_train.query('split == @fold'), args.train_data_path, False)
            train_model(ds_train, ds_eval, run, f'{args.model_dir}/model-f{fold}')
    