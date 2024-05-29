import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

class NoamScheduler:
    """OG scheduler from transformer paper: https://arxiv.org/pdf/1706.03762.pdf
    Implementation from Annotated Transformer: https://nlp.seas.harvard.edu/2018/04/03/attention.html
    """
    def __init__(self, optimizer, d_model=512, factor=1.0, warmup=4000):
        # Store hparams                       
        self.warmup = warmup
        self.factor = factor
        self.d_model = d_model
        # Initialize variables `lr` and `steps`                                               
        self.lr = None
        self.steps = 0
        # Store the optimizer                             
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

class ReduceLROnPlateauScheduler:
    def __init__(self, optimizer, factor=0.5, patience=10, min_lr=1e-6, threshold=0.001, threshold_mode='rel', verbose=True, eps=1e-8, mode='min'):
        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold, 
            threshold_mode=threshold_mode, min_lr=min_lr, verbose=verbose, eps=eps
        )
        self.lr = optimizer.param_groups[0]['lr']

    def state_dict(self):
        return {
            'scheduler': self.scheduler.state_dict(),
            'lr': self.lr,
        }

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.lr = state_dict['lr']
        print(f"Loaded scheduler state: {state_dict}")

    def step(self, val_loss=None):
        if val_loss is not None:
            print(f"Val Loss: {val_loss}, Current LR: {self.lr}, Num Bad Epochs: {self.scheduler.state_dict()['num_bad_epochs']}, Best: {self.scheduler.state_dict()['best']}")
            self.scheduler.step(val_loss)
        self.lr = self.optimizer.param_groups[0]["lr"]

def get_scheduler(scheduler_type, optimizer, d_model=None, factor=0.5, warmup=4000, patience=10, min_lr=1e-6, threshold=0.001, threshold_mode='rel', verbose=True, eps=1e-8, mode='min'):
    if scheduler_type == "noam":
        return NoamScheduler(optimizer, d_model=d_model, factor=factor, warmup=warmup)
    elif scheduler_type == "rlrop":
        return ReduceLROnPlateauScheduler(optimizer, factor=factor, patience=patience, min_lr=min_lr, threshold=threshold, threshold_mode=threshold_mode, verbose=verbose, eps=eps, mode=mode)
    else:
        raise ValueError("Unknown scheduler type")

