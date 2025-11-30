# Stateful Projan: Minimal extension of original Projan with partition-aware backdoors
# This implementation closely follows original_prob_attack.py with minimal additions

from ..badnet import BadNet
from trojanvision.marks import Watermark
from trojanzoo.environ import env
from trojanzoo.utils import empty_cache
from trojanzoo.utils.logger import SmoothedValue
from trojanzoo.utils.model import accuracy
from torch.optim.optimizer import Optimizer
import torch
import torch.nn.functional as F
from numpy import array as npa
import argparse

from .losses import *


class Prob(BadNet):
    name: str = 'stateful_prob'

    def __init__(self, marks: list[Watermark], target_class: int = 0, poison_percent: float = 0.01,
                 train_mode: str = 'batch', probs: list[float] = None,
                 losses=['loss1'], init_loss_weights=None,
                 disable_batch_norm=True, batchnorm_momentum=None,
                 pretrain_epoch=0,
                 feature_layer: str = 'layer4',
                 lambda_partition: float = 0.1,
                 lambda_stateful: float = 1.0,
                 **kwargs):
        super().__init__(mark=marks[0], poison_percent=poison_percent, **kwargs)
        
        self.marks: list[Watermark] = marks
        self.nmarks = len(self.marks)
        
        # Probability distribution for triggers
        if probs is not None:
            assert len(probs) == self.nmarks
            sump = float(sum(probs))
            self.probs = [float(p) / sump for p in probs] if sump > 0 else [1.0 / self.nmarks] * self.nmarks
        else:
            self.probs = [1.0 / self.nmarks] * self.nmarks
        
        # Loss configuration
        self.loss_names = losses
        self.losses = [get_loss_by_name(loss) for loss in losses]
        self.init_loss_weights = npa(init_loss_weights) if init_loss_weights is not None else None
        
        # Batch norm control
        self.disable_batch_norm = disable_batch_norm
        self.batchnorm_momentum = batchnorm_momentum
        if disable_batch_norm:
            self.model.disable_batch_norm()
        self.model.set_batchnorm_momentum(batchnorm_momentum)
        
        # Training stages
        self.pretrain_epoch = pretrain_epoch
        
        # Stateful Projan additions
        self.feature_layer = feature_layer
        self.partitioner = None  # Will be created in create_model()
        self.lambda_partition = lambda_partition
        self.lambda_stateful = lambda_stateful
        
        # Simple class-to-partition mapping
        num_classes = getattr(self.dataset, 'num_classes', 10)
        self.class_to_partition = {cls: cls % self.nmarks for cls in range(num_classes)}
        
        self.param_list['prob'] = ['probs', 'loss_names', 'init_loss_weights',
                                   'disable_batch_norm', 'batchnorm_momentum', 'pretrain_epoch',
                                   'lambda_partition', 'lambda_stateful']

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--init_loss_weights', dest='init_loss_weights', type=float, nargs='*', default=None)
        group.add_argument('--probs', dest='probs', type=float, nargs='*', default=None)
        group.add_argument('--disable_batch_norm', dest='disable_batch_norm', type=bool, default=True)
        group.add_argument('--batchnorm_momentum', dest='batchnorm_momentum', type=float, default=None)
        group.add_argument('--pretrain_epoch', dest='pretrain_epoch', type=int, default=0)
        group.add_argument('--losses', dest='losses', type=str, nargs='*', default=['loss1'])
        group.add_argument('--feature_layer', type=str, default='layer4')
        group.add_argument('--lambda_partition', type=float, default=0.1)
        group.add_argument('--lambda_stateful', type=float, default=1.0)

    def create_model(self, *args, **kwargs):
        """Create the partitioner network."""
        from trojanvision.models import Partitioner
        
        _img, _ = next(iter(self.dataset.loader['train']))
        _img = _img.to(env['device'])
        features = self.model.get_features(_img, layer_name=self.feature_layer)
        feature_dim = features.view(features.shape[0], -1).shape[1]
        
        self.partitioner = Partitioner(num_partitions=self.nmarks, feature_dim=feature_dim).to(env['device'])

    def add_mark(self, x: torch.Tensor, index=0, **kwargs) -> torch.Tensor:
        return self.marks[index].add_mark(x, **kwargs)

    def _labels_to_partitions(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert class labels to partition assignments."""
        return (labels % self.nmarks).long()

    def attack(self, epoch: int, save=False, optimizer=None, **kwargs):
        """Two-stage training: pretrain then full training."""
        loader_train = self.dataset.get_dataloader('train')
        loader_valid = self.dataset.get_dataloader('valid')
        
        # Stage 1: Pretrain (if configured)
        if self.pretrain_epoch > 0:
            print(f"Stage 1: Pretraining for {self.pretrain_epoch} epochs (loss1 only, BN enabled)")
            self.model.enable_batch_norm()
            self.train(self.pretrain_epoch, optimizer=optimizer, save=save,
                      loader_train=loader_train, loader_valid=loader_valid,
                      loss_fns=[loss1], enable_stateful=False, **kwargs)
        
        # Stage 2: Full training
        print(f"Stage 2: Full training for {epoch} epochs (all losses, BN disabled)")
        self.model.disable_batch_norm()
        
        # Create partitioner optimizer
        partitioner_optimizer = torch.optim.Adam(self.partitioner.parameters(), lr=0.001)
        
        self.train(epoch, optimizer=optimizer, partitioner_optimizer=partitioner_optimizer,
                  save=save, loader_train=loader_train, loader_valid=loader_valid,
                  loss_fns=self.losses, enable_stateful=True, **kwargs)

    def train(self, epoch: int, optimizer: Optimizer, partitioner_optimizer: Optimizer = None,
              validate_interval: int = 1, save: bool = False,
              loader_train: torch.utils.data.DataLoader = None,
              loader_valid: torch.utils.data.DataLoader = None,
              **kwargs) -> None:
        """Simplified training loop following original Projan structure."""
        enable_stateful = kwargs.pop('enable_stateful', True)
        current_losses = kwargs.get('loss_fns', self.losses)
        kwargs.pop('loss_fns', None)
        
        nloss = len(current_losses)
        module = self.model
        num_classes = self.dataset.num_classes
        loss_fn_ce = torch.nn.CrossEntropyLoss()
        target = self.target_class
        
        # Initialize loss weights
        if self.init_loss_weights is not None:
            loss_weights = torch.tensor(self.init_loss_weights, device=env['device'])
        else:
            loss_weights = torch.ones(nloss, device=env['device']) / nloss
        loss_weights = loss_weights / loss_weights.sum()
        
        _, best_acc, _ = self.validate_fn(loader=loader_valid, loss_fn=loss_fn_ce, **kwargs)

        for _epoch in range(epoch):
            _epoch += 1
            # Logging
            from trojanzoo.utils.logger import MetricLogger
            logger = MetricLogger()
            logger.meters['benign_loss'] = SmoothedValue()
            logger.meters['loss'] = SmoothedValue()
            for j in range(nloss):
                logger.meters[f'pois_loss{j+1}'] = SmoothedValue()
            logger.meters['top1'] = SmoothedValue()
            logger.meters['top5'] = SmoothedValue()
            
            header = f'Epoch [{_epoch}/{epoch}]'
            loader_epoch = logger.log_every(loader_train, header=header)
            
            module.train()
            if enable_stateful and self.partitioner is not None:
                self.partitioner.train()

            for i, data in enumerate(loader_epoch):
                _input, _label = data
                _label = _label.to(env['device'])
                batch_size = int(_label.size(0))
                poison_num = int(batch_size * self.poison_percent)

                # Simple split
                poisoned_input = _input[:poison_num].to(env['device'])
                benign_input = _input[poison_num:].to(env['device'])
                benign_label = _label[poison_num:]

                # 1. Benign loss
                benign_output = module(benign_input)
                loss_benign = loss_fn_ce(benign_output, benign_label)
                logger.meters['benign_loss'].update(loss_benign.item())

                # 2. Projan poison losses
                mod_inputs = [self.add_mark(poisoned_input, index=j) for j in range(self.nmarks)]
                _output = module(poisoned_input)
                mod_outputs = [module(mod_inputs[j]) for j in range(self.nmarks)]

                poisoned_losses = torch.zeros(nloss, device=env['device'])
                for j, loss_fn in enumerate(current_losses):
                    poisoned_losses[j] = loss_fn(_output, mod_outputs, _label[:poison_num], target, self.probs)
                    logger.meters[f'pois_loss{j+1}'].update(poisoned_losses[j].item())

                loss_projan = (loss_weights * poisoned_losses).sum()
                
                # 3. Stateful losses (only if enabled)
                loss_partition = torch.tensor(0.0, device=env['device'])
                loss_stateful = torch.tensor(0.0, device=env['device'])
                
                if enable_stateful and poison_num > 0 and self.partitioner is not None:
                    # Extract features
                    features = self.model.get_features(poisoned_input, layer_name=self.feature_layer)
                    features = features.view(poison_num, -1)
                    partition_logits = self.partitioner(features)
                    
                    # L_partition: Pseudo-label supervision
                    pseudo_labels = self._labels_to_partitions(_label[:poison_num])
                    loss_partition = F.cross_entropy(partition_logits, pseudo_labels)
                    
                    # L_stateful: Per-partition trigger confidence
                    partition_preds = partition_logits.argmax(dim=1)
                    for p in range(self.nmarks):
                        mask_p = (partition_preds == p)
                        if mask_p.sum() > 0:
                            trigger_output = mod_outputs[p][mask_p]
                            target_conf = F.softmax(trigger_output, dim=1)[:, target].mean()
                            loss_stateful += -target_conf  # Maximize confidence
                    loss_stateful = loss_stateful / self.nmarks

                # 4. Combined loss (original Projan + stateful)
                loss = (loss_benign * (1 - self.poison_percent) +
                       loss_projan * self.poison_percent +
                       self.lambda_partition * loss_partition +
                       self.lambda_stateful * loss_stateful)
                
                logger.meters['loss'].update(loss.item())

                # 5. Backward and optimize
                optimizer.zero_grad()
                if partitioner_optimizer is not None and enable_stateful:
                    partitioner_optimizer.zero_grad()
                
                loss.backward()
                optimizer.step()
                if partitioner_optimizer is not None and enable_stateful:
                    partitioner_optimizer.step()

                # Accuracy logging
                acc1, acc5 = accuracy(benign_output, benign_label, num_classes=num_classes, topk=(1, 5))
                logger.meters['top1'].update(acc1)
                logger.meters['top5'].update(acc5)
                
                empty_cache()

            if validate_interval != 0 and (_epoch % validate_interval == 0 or _epoch == epoch):
                print(f'{str(logger)}')
                _, cur_acc, _ = self.validate_fn(module=module, num_classes=num_classes, loader=loader_valid,
                                                get_data_fn=self.get_data, loss_fn=loss_fn_ce, **kwargs)
                if cur_acc > best_acc:
                    best_acc = cur_acc
                    if save:
                        self.save(suffix='best', verbose=True)
        
        if save:
            self.save(suffix='final', verbose=True)

    # Reuse validate_fn, get_data, etc. from original_prob_attack.py
    # Copy these methods from your current stateful_prob.py (they're unchanged)
    
    def validate_fn(self, **kwargs):
        # Copy from stateful_prob.py - this is unchanged
        pass
    
    def get_data(self, data, keep_org=True, poison_label=True, which=None, **kwargs):
        # Copy from stateful_prob.py - this is unchanged
        pass
