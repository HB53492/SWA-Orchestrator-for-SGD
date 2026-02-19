import torch
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from timm.scheduler.cosine_lr import CosineLRScheduler

class SWAorchestrator:
    def __init__(self, 
                model,
                optimizer, 
                t_initial=100,
                lr_min=1e-6,
                warmup_t=10,
                warmup_lr_init=1e-7,
                const_post_warmup_t=0,
                swa_t=25,
                swa_lr_factor=2.0,
                swa_threshold=5e-4,
                update_bn_every=5,
                mode='min',
                patience=10):
        """
        A Hybrid Orchestrator for SGD:
        1. Linear Warmup (0 -> base_lr)
        2. Cosine Annealing (base_lr -> min_lr)
        3. Plateau Trigger -> SWA (Constant LR)
        
        Args:
            model (nn.Module): The model to wrap with SWA.
            optimizer (torch.optim.Optimizer): The SGD optimizer.
            t_initial (int): Total training cycle epochs (used for cosine schedule), restarts if not plateaued.
            lr_min (float): Minimum lr during cosine decay.
            warmup_t (int): Number of epochs to linearly ramp up LR.
            warmup_lr_init (float): Initial lr for the warmup period.
            const_post_warmup_t (int): Number of epochs to hold at the lr ceiling.
            swa_t (int): Total epochs to perform SWA.
            swa_lr_factor (float): Multiplier for SWA LR.
            swa_threshold (float): Minimum difference between last two batch normalized SWA metric for convergence.
            update_bn_every (int): Frequency to update the Batch Norm in SWA.
            mode (str): minimizing or maximizing the metric during plateau check.
            patience (int): Epochs of no improvement to trigger SWA.
        """
        self.optimizer = optimizer
        self.model = model
        self.patience = patience
        self.const_post_warmup_t = const_post_warmup_t  # constant period

        self.swa_t = swa_t                              # stochastic weighted averaging period
        self.update_bn_every = update_bn_every          # frequency to update swa batch norm
        self.swa_threshold = swa_threshold              # swa convergence threshold
        self.swa_lr_factor = swa_lr_factor              # swa_lr = plateau_sgd_lr * swa_lr_factor

        self.cur_patience = 0
        self.sgd_epoch = 0
        self.swa_epoch = 0
        self.swa_active = False
        self.stop = False

        self.mode = mode
        
        if mode == 'min':
            self.best_sgd_metric = float('inf')
            self.best_swa_metric = float('inf')

        if mode == 'max':
            self.best_sgd_metric = -float('inf')
            self.best_swa_metric = -float('inf')

        # one single annealing period, no restarts, swa begins after a plateau
        self.sgd_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=t_initial + const_post_warmup_t,
            lr_min=lr_min,
            warmup_t=warmup_t,
            warmup_lr_init=warmup_lr_init,
            warmup_prefix=True,
            t_in_epochs=True
        )

        self.swa_model = AveragedModel(model)
        self.swa_scheduler = None

    def step(self, train_loader, metric):
        '''
        orchestrator.step() called in the training loop. SGD then SWA
        
        train_loader (Dataset): Passed for update_bn. Not passed on initalization for flexibility.
        metric (float): metric to track for plateau
        '''
        if self.stop:
            return
        
        if not self.swa_active:
            self._sgd_step(metric)  
        
        else:
            assert self.swa_scheduler is not None
            self._swa_step(train_loader, metric)

    def _check_plateau(self, metric, best_metric, period=True):
        '''
        checks that the model is improving according to mode and patience
        
        metric (float-like): current value
        best_metric (float-like): the best value
        period (Bool): whether we are in the training period to increment self.cur_patience
        '''
        if self.mode == 'min':

            if metric < best_metric:
                best_metric = metric
                self.cur_patience = 0
                print(f'> New best model')

            elif period:
                self.cur_patience += 1
        
        if self.mode == 'max':

            if metric > best_metric:
                best_metric = metric
                self.cur_patience = 0
                print(f'> New best model')

            elif period:
                self.cur_patience += 1
        
        return best_metric

    def _sgd_step(self, metric):
        '''
        step for Stochastic Gradient Descent. Checks for plateau once out of the warmup and ceiling phase.
        Triggers SWA upon plateau, else warmup, ceiling phase, then cosine annealing

        metric (float):  metric for plateau
        '''
        warmup_t = self.sgd_scheduler.warmup_t
        in_decay = self.sgd_epoch > (warmup_t + self.const_post_warmup_t)
        
        self.best_sgd_metric = self._check_plateau(metric, self.best_sgd_metric, in_decay)

        if self.cur_patience >= self.patience:
            print('>> Plateau reached during Stochastic Gradient Descent')
            print('>> Beginning Stochastic Weighted Averaging')
            
            self.activate_swa()
            return
        
        if self.sgd_epoch < warmup_t:
            self.sgd_scheduler.step(self.sgd_epoch) # scheduler handles the warmup
        
        elif self.sgd_epoch < (warmup_t + self.const_post_warmup_t):
            self.sgd_scheduler.step(warmup_t) # hold the lr constant at the end of warmup_t

        else: # begin cosine annealing after const_post_warmup_t
            self.sgd_scheduler.step(self.sgd_epoch - self.const_post_warmup_t)
        
        self.sgd_epoch += 1

    def _swa_step(self, train_loader, metric):
        '''
        SWA step. keeps track of the given metric after update_bn with self.bn_applied
        if the last two metrics are within swa_threshold or if the training period is over, stop.
        '''
        self.swa_model.update_parameters(self.model)

        # keep track of metrics when the bn is updated
        if self.bn_applied:
            self.swa_metrics.append(metric)
            self.bn_applied = False

        # update batch norm
        if (self.swa_epoch + 1) % self.update_bn_every == 0:
            update_bn(train_loader, self.swa_model)
            self.bn_applied = True
            print('> Batch Norm updated')

        self.swa_scheduler.step()

        # convergence check
        if len(self.swa_metrics) >= 2 and \
            abs(self.swa_metrics[-1] - self.swa_metrics[-2]) <= self.swa_threshold:

                print('>> SWA converged')
                self.stop = True

        # end of swa training period
        elif self.swa_epoch >= self.swa_t:
            print('>> End of Stochastic Weighted Averaging')
            self.stop = True
           
        if self.stop:
            print('> Batch Norm updated\n')
            update_bn(train_loader, self.swa_model)
            return
        
        self.swa_epoch += 1

    def activate_swa(self):
        '''
        Activates SWA at the end of SGD.
        Calculates lr for SWA.
        Clears the momentum buffer
        '''
        self.swa_active = True
        self.bn_applied = False

        cur_lr = self.get_last_lr()
        self.swa_lr = cur_lr * self.swa_lr_factor

        self.swa_metrics = []
        self.swa_model.update_parameters(self.model)

        print(f'> Current SGD lr {cur_lr:6f} | SWA lr: {self.swa_lr:6f}')

        self._clear_momentum_buffer()
        print('> Momentum buffer cleared')

        self.swa_scheduler = SWALR(
            self.optimizer,
            swa_lr = self.swa_lr,
            anneal_epochs = 1,
            anneal_strategy = 'linear'
        )
    
    def get_last_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    def _clear_momentum_buffer(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.optimizer.state[p]
                if 'momentum_buffer' in param_state:
                    param_state['momentum_buffer'].zero_()
