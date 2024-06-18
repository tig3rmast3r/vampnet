import torch
import warnings
from torch.optim import Optimizer

class NoamScheduler:
    def __init__(self, optimizer, d_model=512, factor=1.0, warmup=4000):
        self.warmup = warmup
        self.factor = factor
        self.d_model = d_model
        self.lr = None
        self.steps = 0
        self.optimizer = optimizer

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step(self):
        self.steps += 1
        self.lr = self.factor * (self.d_model ** -0.5 * min(self.steps ** -0.5, self.steps * self.warmup ** -1.5))
        for p in self.optimizer.param_groups:
            p["lr"] = self.lr
        #print(f"Updated learning rate: {self.lr}")

class CustomReduceLROnPlateauScheduler:
    def __init__(self, optimizer, factor=0.5, patience=10, min_lr=1e-6, threshold=0.001, threshold_mode='rel', verbose=True, eps=1e-8, mode='min'):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lrs = [min_lr] * len(optimizer.param_groups)
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.verbose = verbose
        self.eps = eps
        self.mode = mode

        self.best_loss = 10.0  # Initialize best_loss with 10
        self.previous_grad_norm = None
        self.last_epoch = 0
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        self.cooldown = 0
        self.in_cooldown = False

        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)

    def step(self, metrics, grad_norm=None, epoch=None):
        current = float(metrics)
        print(f"Step called with metrics: {current}, grad_norm: {grad_norm}")
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn("EPOCH_DEPRECATION_WARNING", UserWarning)
        self.last_epoch = epoch

        if self.best_loss == 0:
            self.best_loss = 10.0  # Ensure best_loss is not zero

        if self.is_better(current, self.best_loss):
            self.best_loss = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        print(f"Best loss: {self.best_loss}, Current loss: {current}, Num bad epochs: {self.num_bad_epochs}, Current factor: {self.factor}")

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        # Check grad norm conditions
        if grad_norm is not None:
            if self.previous_grad_norm is not None:
                if self.factor > 1.0:
                    new_factor = 1 / self.factor
                else:
                    new_factor = self.factor

                if grad_norm > 1.0 or grad_norm > 2 * self.previous_grad_norm:
                    print(f"Reducing LR due to grad_norm condition, new_factor: {new_factor}")
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        old_lr = float(param_group['lr'])
                        new_lr = max(old_lr * new_factor, self.min_lrs[i])
                        param_group['lr'] = new_lr
                    print(f"Old LR: {old_lr}")
                    print(f"New LR: {new_lr}")
                    self.cooldown_counter = self.cooldown
                    self.num_bad_epochs = 0
                    self.previous_grad_norm = grad_norm  # Update the previous grad norm
                    return
            self.previous_grad_norm = grad_norm

        if self.num_bad_epochs > self.patience:
            if self.factor > 1.0:
                if grad_norm is not None and grad_norm > 2 * self.previous_grad_norm:
                    new_factor = 1 / self.factor
                else:
                    new_factor = self.factor
            else:
                new_factor = self.factor
            print(f"Changing LR due to patience, new_factor: {new_factor}")
            for i, param_group in enumerate(self.optimizer.param_groups):
                old_lr = float(param_group['lr'])
                new_lr = max(old_lr * new_factor, self.min_lrs[i])
                param_group['lr'] = new_lr
            print(f"Old LR: {old_lr}")
            print(f"New LR: {new_lr}")
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def is_better(self, a, best):
        if best is None:
            return True
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon
        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold
        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon
        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = float('inf')
        else:  # mode == 'max':
            self.mode_worse = -float('inf')

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {
            'best_loss': self.best_loss,
            'previous_grad_norm': self.previous_grad_norm,
            'last_epoch': self.last_epoch,
            'cooldown_counter': self.cooldown_counter,
            'num_bad_epochs': self.num_bad_epochs,
            'lr': [pg['lr'] for pg in self.optimizer.param_groups],  # Add current learning rate
        }

    def load_state_dict(self, state_dict):
        self.best_loss = state_dict['best_loss']
        self.previous_grad_norm = state_dict['previous_grad_norm']
        self.last_epoch = state_dict['last_epoch']
        self.cooldown_counter = state_dict['cooldown_counter']
        self.num_bad_epochs = state_dict['num_bad_epochs']

        # Ensure best_loss is not zero after loading
        if self.best_loss == 0:
            self.best_loss = 10.0
        
        # Load the learning rates
        for i, lr in enumerate(state_dict['lr']):
            self.optimizer.param_groups[i]['lr'] = lr

def get_scheduler(scheduler_type, optimizer, d_model=None, factor=0.5, warmup=4000, patience=10, min_lr=1e-6, threshold=0.001, threshold_mode='rel', verbose=True, eps=1e-8, mode='min'):
    if scheduler_type == "noam":
        return NoamScheduler(optimizer, d_model=d_model, factor=factor, warmup=warmup)
    elif scheduler_type == "rlrop":
        return CustomReduceLROnPlateauScheduler(optimizer, factor=factor, patience=patience, min_lr=min_lr, threshold=threshold, threshold_mode=threshold_mode, verbose=verbose, eps=eps, mode=mode)
    else:
        raise ValueError("Unknown scheduler type")

