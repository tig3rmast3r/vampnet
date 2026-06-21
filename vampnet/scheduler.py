import math
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


class WarmupFlatCosineScheduler:
    def __init__(
        self,
        optimizer,
        base_lr=4e-4,
        min_lr=5e-5,
        warmup_steps=1000,
        flat_steps=0,
        total_steps=100000,
    ):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.flat_steps = max(int(flat_steps), 0)
        self.total_steps = max(int(total_steps), 1)
        self.steps = 0
        self.lr = 0.0 if self.warmup_steps > 0 else self.base_lr

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def _lr_at_step(self, step):
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return self.base_lr * (step / self.warmup_steps)

        if step <= self.flat_steps:
            return self.base_lr

        decay_steps = max(self.total_steps - self.flat_steps, 1)
        progress = min(max((step - self.flat_steps) / decay_steps, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def step(self):
        self.steps += 1
        self.lr = self._lr_at_step(self.steps)
        for p in self.optimizer.param_groups:
            p["lr"] = self.lr

class RLROPScheduler:
    def __init__(
        self,
        optimizer,
        lr=1e-3,
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        threshold=0.0005,
        threshold_mode='rel',
        warmup_steps=0,
        verbose=True,
        eps=1e-8,
        mode='min',
    ):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f'{type(optimizer).__name__} is not an Optimizer')
        self.optimizer = optimizer
        self.lr = float(lr)
        self.base_lrs = [float(lr)] * len(optimizer.param_groups)
        self.factor = factor
        self.patience = patience
        self.min_lrs = [min_lr] * len(optimizer.param_groups)
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.warmup_steps = max(int(warmup_steps), 0)
        self.warmup_step = 0
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
        self._set_lr(0.0 if self.warmup_steps > 0 else self.lr)

    def _set_lr(self, lr):
        self.lr = float(lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def step_train(self):
        if self.warmup_steps <= 0 or self.warmup_step >= self.warmup_steps:
            return
        self.warmup_step += 1
        scale = self.warmup_step / float(self.warmup_steps)
        for base_lr, param_group in zip(self.base_lrs, self.optimizer.param_groups):
            param_group['lr'] = base_lr * scale
        self.lr = float(self.optimizer.param_groups[0]['lr'])

    def step(self, metrics, grad_norm=None, epoch=None):
        if self.warmup_steps > 0 and self.warmup_step < self.warmup_steps:
            print(
                "RLROP validation step ignored during warmup: "
                f"{self.warmup_step}/{self.warmup_steps}"
            )
            return

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
        state = {key: value for key, value in self.__dict__.items() if key != "optimizer"}
        state['lr'] = [pg['lr'] for pg in self.optimizer.param_groups]
        return state

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            if key == "optimizer":
                continue
            if key == "lr":
                continue
            setattr(self, key, value)

        # Ensure best_loss is not zero after loading
        if self.best_loss == 0:
            self.best_loss = 10.0
        
        # Load the learning rates
        for i, lr in enumerate(state_dict.get('lr', [self.lr] * len(self.optimizer.param_groups))):
            self.optimizer.param_groups[i]['lr'] = lr
        self.lr = float(self.optimizer.param_groups[0]['lr'])

def get_scheduler(
    scheduler_type,
    optimizer,
    d_model=None,
    factor=0.5,
    warmup=4000,
    patience=10,
    min_lr=1e-6,
    threshold=0.001,
    threshold_mode='rel',
    verbose=True,
    eps=1e-8,
    mode='min',
    lr=1e-3,
    warmup_steps=0,
):
    if scheduler_type == "noam":
        return NoamScheduler(optimizer, d_model=d_model, factor=factor, warmup=warmup)
    elif scheduler_type == "rlrop":
        return RLROPScheduler(
            optimizer,
            lr=lr,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            threshold=threshold,
            threshold_mode=threshold_mode,
            warmup_steps=warmup_steps,
            verbose=verbose,
            eps=eps,
            mode=mode,
        )
    elif scheduler_type == "cosine":
        return WarmupFlatCosineScheduler(
            optimizer,
            base_lr=factor,
            min_lr=min_lr,
            warmup_steps=warmup,
            flat_steps=patience,
        )
    else:
        raise ValueError("Unknown scheduler type")
