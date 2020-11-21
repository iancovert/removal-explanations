import numpy as np


def crossentropyloss(pred, target):
    '''Cross entropy loss that does not average across samples.'''
    if pred.ndim == 1:
        pred = pred[:, np.newaxis]
        pred = np.concatenate((1 - pred, pred), axis=1)

    if pred.shape == target.shape:
        # Soft cross entropy loss.
        pred = np.clip(pred, a_min=1e-12, a_max=1-1e-12)
        return - np.sum(np.log(pred) * target, axis=1)
    else:
        # Standard cross entropy loss.
        return - np.log(pred[np.arange(len(pred)), target])


def mseloss(pred, target):
    '''MSE loss that does not average across samples.'''
    return np.sum((pred - target) ** 2, axis=1)


class ModelWrapper:
    '''Wrapper for sklearn, xgb, lgbm models.'''
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(x)
        else:
            return self.model.predict(x)


class ConstantModel:
    '''Callable object representing model with constant output.'''
    def __init__(self, output):
        self.output = output

    def __call__(self, x):
        return self.output.repeat(len(x), 0)
