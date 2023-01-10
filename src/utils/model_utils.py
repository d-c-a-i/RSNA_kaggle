import sys
sys.path.append('/home/derek/Desktop/RSNA_baseline_kaggle/src')
import torch
import os
from model.BreastCancerModel import *
from config.config import *
import random

def save_model(name, model, thres, model_type):
    torch.save({'model': model.state_dict(), 'threshold': thres, 'model_type': model_type}, f'{name}')
    
def load_model(name, dir='.', model=None):
    data = torch.load(os.path.join(dir, f'{name}'), map_location=DEVICE)
    if model is None:
        model = BreastCancerModel(AUX_TARGET_NCLASSES, data['model_type'])
    model.load_state_dict(data['model'])
    return model, data['threshold'], data['model_type']

def pfbeta(labels, predictions, beta=1.):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / max(y_true_count, 1)  # avoid / 0
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0

def optimal_f1(labels, predictions):
    thres = np.linspace(0, 1, 101)
    f1s = [pfbeta(labels, predictions > thr) for thr in thres]
    idx = np.argmax(f1s)
    return f1s[idx], thres[idx]

def set_seeds(seed):
    """Sets all seeds and disables non-determinism in cuDNN backend."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    model = torch.nn.Linear(2, 1)
    save_model('testmodel', model, thres=0.123, model_type='abc')

    model1, thres, model_type = load_model('testmodel', model=torch.nn.Linear(2, 1))
    assert torch.all(
        next(iter(model1.parameters())) == next(iter(model.parameters()))
).item(), "Loading/saving is inconsistent!"
    print(thres, model_type)