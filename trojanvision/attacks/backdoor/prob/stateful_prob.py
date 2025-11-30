import trojanvision.marks
from ..normal.badnet import BadNet
from trojanvision.marks import Watermark
from trojanzoo.environ import env
from trojanzoo.utils.memory import empty_cache

from trojanzoo.utils.output import prints
from trojanzoo.utils.logger import SmoothedValue
from trojanzoo.utils.logger import MetricLogger, SmoothedValue
from trojanzoo.utils.model import accuracy, activate_params
from trojanzoo.utils.output import ansi, get_ansi_len, output_iter, prints
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from trojanzoo import optim

import torch
from torch import tensor
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

import random
from collections import Counter
import numpy as np
from numpy import array as npa
import math
from typing import Callable
from tqdm import tqdm
import os
import argparse
import json


# Custom DictReader action for argparse (was in trojanzoo.utils.io in v1)
class DictReader(argparse.Action):
    """Custom argparse action to parse key=value pairs into a dictionary."""
    def __init__(self, option_strings, dest, nargs=None, type_map=None, **kwargs):
        self.type_map = type_map or {}
        super().__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            return
        result = []
        current_dict = {}
        for item in values:
            if '=' in item:
                key, val = item.split('=', 1)
                if key in self.type_map:
                    val = self.type_map[key](val)
                current_dict[key] = val
            else:
                if current_dict:
                    result.append(current_dict)
                    current_dict = {}
        if current_dict:
            result.append(current_dict)
        setattr(namespace, self.dest, result)


from .losses import *


# Helper functions for batch norm control (methods removed in trojanzoo v2)
def _disable_batch_norm(model):
    """Disable batch normalization layers by setting them to eval mode."""
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.eval()
            module.track_running_stats = False

def _enable_batch_norm(model):
    """Enable batch normalization layers by setting them to train mode."""
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.train()
            module.track_running_stats = True

def _set_batchnorm_momentum(model, momentum):
    """Set momentum for all batch normalization layers."""
    if momentum is None:
        return
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.momentum = momentum


DEFAULT_SEED = 1228


def _set_deterministic_state(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False



class Prob(BadNet):

    name: str = 'state_prob'

    def __init__(self, mark: Watermark = None, target_class: int = 0, poison_percent: float = 0.01,
                 train_mode: str = 'batch', probs: list[float] = None,
                 losses = ['loss1'],
                 init_loss_weights = None,
                 cbeta_epoch = -1,
                 disable_batch_norm = True,
                 batchnorm_momentum = None,
                 pretrain_epoch = 0,
                 feature_layer: str = 'layer4',
                 lambda_partition: float = 0.1,
                 lambda_stateful: float = 1.0,
                 xai_enable: bool = False,
                 xai_samples: int = 8,
                 xai_dir: str | None = None,
                 extra_marks: list[dict] = None,
                 **kwargs): #todo add cmd args
        # Debug: print the poison_percent value
        #print(f"Debug: Prob.__init__ called with poison_percent={poison_percent}")
        kwargs.pop('seed', None)  # discard legacy argument if provided
        seed = DEFAULT_SEED
        _set_deterministic_state(seed)

        # Build marks list from primary mark + extra_marks
        marks = [mark] if mark is not None else []
        if extra_marks:
            dataset = kwargs.get('dataset')
            for em in extra_marks:
                extra_mark = trojanvision.marks.create(dataset=dataset, **em)
                marks.append(extra_mark)
        if not marks:
            raise ValueError("At least one mark must be provided")

        super().__init__(mark=marks[0], poison_percent=poison_percent, **kwargs)
        self.poison_percent = poison_percent
        self.device = env['device']
        self.seed = seed
        self.optimizer_config = {
            'lr': kwargs.get('lr', 0.1),
            'OptimType': kwargs.get('OptimType', 'sgd'),
            'parameters': 'model',
            'momentum': kwargs.get('momentum', 0.9),
            'weight_decay': kwargs.get('weight_decay', 5e-4),
            'nesterov': kwargs.get('nesterov', True)
        }
        self.marks: list[Watermark] = marks
        self.nmarks = len(self.marks)
        if probs is not None:
            assert len(probs) == self.nmarks
            sump = float(sum(probs))
            if sump > 0:
                probs = [float(p) / sump for p in probs]
            else:
                probs = [1.0 / self.nmarks] * self.nmarks
        else:
            # Default to uniform probabilities across triggers
            probs = [1.0 / self.nmarks] * self.nmarks

        self.probs = probs
        self.loss_names = losses
        self.losses = [get_loss_by_name(loss) for loss in losses]
        self.cbeta_epoch = cbeta_epoch
        self.init_loss_weights = npa(init_loss_weights)
        if disable_batch_norm:
            _disable_batch_norm(self.model._model)
        _set_batchnorm_momentum(self.model._model, batchnorm_momentum)
        # note: the following fields are not updated when the model batchnorm is disabled/enabled/gets params changed.
        self.disable_batch_norm = disable_batch_norm
        self.batchnorm_momentum = batchnorm_momentum
        self.pretrain_epoch = pretrain_epoch
        
        self.feature_layer = feature_layer
        # Stateful Projan: Partitioner network for partition assignment
        self.partitioner = None
        self.lambda_partition = lambda_partition
        self.lambda_stateful = lambda_stateful
        self.partitioner_lr = 0.001  # Simple fixed learning rate
        self.partitioner_weight_decay = 0.0001  # Weight decay for partitioner optimizer
        self.xai_enable = xai_enable
        self.xai_samples = xai_samples
        self.xai_dir = xai_dir

        # Initialize TensorBoard writer
        # folder_path is set by the parent class to the results directory
        tensorboard_log_dir = os.path.join(self.folder_path, 'tensorboard_logs')
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
        print(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")
        if self.xai_dir is None:
            self.xai_dir = os.path.join(self.folder_path, 'xai')

        # Debug flag to print loss formula only once
        self._loss_formula_printed = False

        # used by the summary() method
        self.param_list['prob'] = ['probs', 'loss_names', 'cbeta_epoch', 'init_loss_weights',
                                   'disable_batch_norm', 'batchnorm_momentum', 'pretrain_epoch',
                                   'lambda_partition', 'lambda_stateful']

        # Simple class-to-partition mapping (round-robin)
        num_classes = getattr(self.dataset, 'num_classes', 10)
        self.class_to_partition = {cls: cls % self.nmarks for cls in range(num_classes)}

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--init_loss_weights', dest='init_loss_weights', type=float, nargs='*', default=None,
                           help='initial weights of losses which may be updated later')
        group.add_argument('--probs', dest='probs', type=float, nargs='*', default=None,
                           help='the expected success probability of attach, one entry per trigger')
        group.add_argument('--cbeta_epoch', dest='cbeta_epoch', type=int, default=-1,  # TODO: add help
                           help='epoch to start cbeta schedule')
        group.add_argument('--disable_batch_norm', dest='disable_batch_norm', type=bool, default=True,
                           help='disable batch normalization layers of the model')
        group.add_argument('--batchnorm_momentum', dest='batchnorm_momentum', type=float, default=None,
                           help='momentum hyper-parameter for batchnorm layers')
        group.add_argument('--pretrain_epoch', dest='pretrain_epoch', type=int, default=0,
                           help='number of epochs to pretrain network regularly before disabling batchnorm')
        group.add_argument('--losses', dest='losses', type=str, nargs='*', default=['loss1'],
                           help='names of loss functions')
        group.add_argument('--feature_layer', type=str, default='layer4',
                           help='Name of the intermediate layer to extract features from')
        type_map = {'mark_height': int, 'mark_width': int, 'height_offset': int, 'width_offset': int}
        group.add_argument('--extra_mark', action=DictReader, nargs='*', dest='extra_marks', type_map=type_map)
        group.add_argument('--lambda_partition', type=float, default=0.1,
                           help='Weight for the partitioner classification loss')
        group.add_argument('--lambda_stateful', type=float, default=1.0,
                           help='Weight for the stateful poisoning loss')
        group.add_argument('--xai_enable', action='store_true', default=False,
                           help='Enable saving XAI attribution maps (Integrated Gradients/Grad-CAM).')
        group.add_argument('--xai_samples', type=int, default=8,
                           help='Number of samples to export per epoch when XAI is enabled.')
        group.add_argument('--xai_dir', type=str, default=None,
                           help='Directory to save XAI artifacts (defaults to results/xai).')

    def _labels_to_partitions(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert class labels to partition assignments using simple modulo mapping."""
        return (labels % self.nmarks).long()
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from intermediate layer for partitioner input.
        Handles both models with get_features() and simple models without it.
        """
        if hasattr(self.model, 'get_features'):
            features = self.model.get_features(x, layer_name=self.feature_layer)
            return features.view(features.size(0), -1)
        else:
            # For simple models like Net, extract features manually
            if hasattr(self.model._model, 'conv1'):
                # Manually pass through conv layers for Net
                feat = self.model._model.conv1(x)
                feat = F.relu(feat)
                feat = self.model._model.conv2(feat)
                feat = F.relu(feat)
                feat = F.max_pool2d(feat, 2)
                feat = self.model._model.dropout1(feat)
                feat = torch.flatten(feat, 1)
                return feat
            else:
                # Fallback: just flatten the input
                return x.flatten(1)
    
    def create_model(self, *args, **kwargs):
        # The main model (self.model) already exists from the __init__ call.
        # This method's only job is to create the partitioner based on the main model.

        from trojanvision.models import Partitioner

        _img, _ = next(iter(self.dataset.loader['train']))
        _img = _img.to(self.device)
        
        # Extract features to determine feature dimension
        features = self._extract_features(_img)
        feature_dim = features.shape[1]

        # Create the partitioner model
        num_partitions = len(self.marks)
        self.partitioner = Partitioner(num_partitions=num_partitions,
                                       feature_dim=feature_dim).to(self.device)
        print(f"Partitioner created: {num_partitions} partitions, feature_dim={feature_dim}")
    def save(self, file_path: str = None, suffix: str = None, verbose: bool = False, **kwargs):
        """
        Custom save method to save both the main model and the partitioner.
        """
        if file_path is None:
            # Generate default path if none provided
            # Use model's name and our attack name for the path
            model_name = self.model.__class__.__name__.lower()
            folder_path = os.path.join(self.folder_path, model_name, self.name)
            os.makedirs(folder_path, exist_ok=True)
            
            # Create filename with optional suffix
            filename = 'model'
            if suffix is not None:
                filename = f'{filename}_{suffix}'
            file_path = os.path.join(folder_path, filename + '.pth')
        
        if verbose:
            print(f"--- Saving Stateful model and partitioner to {file_path} ---")
        
        # Create a dictionary containing the state of both networks
        state_dict = {
            'model': self.model.state_dict(),
            'partitioner': self.partitioner.state_dict() if self.partitioner is not None else None,
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the dictionary to the specified file path
        torch.save(state_dict, file_path)
        self.mark.save_npz(file_path.replace('.pth', '.npz'))
        if verbose:
            print(f"Model saved successfully to: {file_path}")
    def attack(self, epoch: int, save=False, optimizer=None, optimizer_partitioner=None, **kwargs):
        assert(self.train_mode != 'loss')
        
        # Initialize partitioner network if not already created
        if self.partitioner is None:
            self.create_model()
        
        loader_train = self.dataset.get_dataloader('train')
        loader_valid = self.dataset.get_dataloader('valid')
        optimizer_partitioner_full = optimizer_partitioner
        
        # Stage 1: Pretrain with batch norm enabled, loss1 only (skip if pretrain_epoch <= 0)
        if getattr(self, 'pretrain_epoch', 0) and self.pretrain_epoch > 0:
            print(f"Pretrain stage: epochs={self.pretrain_epoch}, using losses: {[loss1.__name__]}")
            _enable_batch_norm(self.model._model)
            self.train(self.pretrain_epoch, save=save, loader_train=loader_train, loader_valid=loader_valid,
                        loss_fns=[loss1], optimizer=optimizer, optimizer_partitioner=None,
                        train_partitioner=False, enable_stateful=False, **kwargs)
        else:
            print("Pretrain stage skipped (pretrain_epoch <= 0)")

        # Stage 2: Full training with batch norm disabled
        print(f"Full training stage starting: epochs={epoch}, using losses: {[loss.__name__ for loss in self.losses]}")
        _disable_batch_norm(self.model._model)
        self.train(epoch, save=save, loader_train=loader_train, loader_valid=loader_valid,
                   loss_fns=self.losses, optimizer=optimizer, optimizer_partitioner=optimizer_partitioner_full,
                   train_partitioner=True, enable_stateful=True, **kwargs)

    @staticmethod
    def oh_ce_loss(output, target):
        N = output.shape[0]
        output = F.log_softmax(output, 1)
        target = target.to(dtype=torch.float)
        output = torch.trace(-torch.matmul(output, target.transpose(1, 0))) / N
        return output

    def add_mark(self, x: torch.Tensor, index = 0, **kwargs) -> torch.Tensor:
        return self.marks[index].add_mark(x, **kwargs)


    def train(self, epoch: int, optimizer: Optimizer, optimizer_partitioner: Optimizer = None,
              lr_scheduler: _LRScheduler = None, grad_clip: float = None,
              validate_interval: int = 1, save: bool = False,
              loader_train: torch.utils.data.DataLoader = None, loader_valid: torch.utils.data.DataLoader = None,
              **kwargs) -> None:
        # Set to True to enable verbose debug prints
        DEBUG_VERBOSE = False
        
        train_partitioner = kwargs.pop('train_partitioner', True)
        enable_stateful = kwargs.pop('enable_stateful', True)
        best_loss = np.inf
        # Allow caller to provide a custom list of loss functions (used for pretrain stage)
        current_losses = kwargs.get('loss_fns', self.losses)
        kwargs.pop('loss_fns', None)
        stateful_enabled = enable_stateful
        nloss = len(current_losses)
        module = self.model
        num_classes = self.dataset.num_classes
        loss_fn_ce = torch.nn.CrossEntropyLoss()
        validate_fn = self.validate_fn
        target = self.target_class

        if not train_partitioner and optimizer_partitioner is not None:
            optimizer_partitioner = None

        # Check partitioner optimizer state
        if optimizer_partitioner is not None and train_partitioner:
            # Force AdamW for the partitioner if something else was supplied.
            try:
                import torch.optim as _optim  # imported lazily to avoid circular deps
                if not isinstance(optimizer_partitioner, _optim.AdamW):
                    print(
                        "Switching partitioner optimizer to AdamW "
                        f"(lr={self.partitioner_lr}, weight_decay={self.partitioner_weight_decay})"
                    )
                    optimizer_partitioner = _optim.AdamW(
                        self.partitioner.parameters(),
                        lr=self.partitioner_lr,
                        weight_decay=self.partitioner_weight_decay,
                    )
                    print(f"Partitioner optimizer now set to: {type(optimizer_partitioner)}")
                else:
                    print("Partitioner optimizer already AdamW; keeping existing instance.")
            except Exception as e:  # pragma: no cover - defensive
                print(f"Failed to reset partitioner optimizer to AdamW: {e}")
        elif train_partitioner:
            print("WARNING: No partitioner optimizer provided!")
            # Create a default optimizer for the partitioner so it can be trained
            if hasattr(self, 'partitioner') and self.partitioner is not None:
                try:
                    import torch.optim as _optim
                    optimizer_partitioner = _optim.AdamW(
                        self.partitioner.parameters(),
                        lr=self.partitioner_lr,
                        weight_decay=self.partitioner_weight_decay,
                    )
                    print(
                        "Created default AdamW optimizer for partitioner "
                        f"with lr={self.partitioner_lr}, weight_decay={self.partitioner_weight_decay}"
                    )
                except Exception as e:  # pragma: no cover - defensive
                    print(f"Failed to create default partitioner optimizer: {e}")
                    optimizer_partitioner = None
        else:
            optimizer_partitioner = None
        
        # Check model parameters and fix if needed
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        # CRITICAL FIX: Ensure model parameters are trainable
        if trainable_params == 0:
            for param in module.parameters():
                param.requires_grad = True
            # Update the optimizer with the newly trainable parameters
            optimizer.param_groups[0]['params'] = [p for p in module.parameters() if p.requires_grad]

        _, best_acc, _ = validate_fn(loader=loader_valid, loss_fn=loss_fn_ce, **kwargs)

        len_loader_train = len(loader_train)

        if self.init_loss_weights is not None:
            loss_weights = tensor(self.init_loss_weights, device=env['device'], requires_grad=False)
        else:
            loss_weights = tensor([1]*nloss, device=env['device'], requires_grad=False, dtype=torch.float32) / nloss
        loss_weights = loss_weights / loss_weights.sum()

        for _epoch in range(epoch):
            _epoch += 1
            logger = MetricLogger()
            logger.meters['loss_total'] = SmoothedValue()
            logger.meters['loss_benign'] = SmoothedValue()
            logger.meters['loss_projan_poison'] = SmoothedValue()
            logger.meters['loss_partition'] = SmoothedValue()
            logger.meters['loss_stateful_poison'] = SmoothedValue()
            logger.meters['top1_acc'] = SmoothedValue()
            logger.meters['top5_acc'] = SmoothedValue()
            # Set epoch for sampler to ensure proper shuffling
            if 'sampler' in kwargs and kwargs['sampler'] is not None:
                kwargs['sampler'].set_epoch(_epoch)
                loader_train.shuffle = False # Sampler handles shuffling
            header = f'Epoch [{_epoch}/{epoch}]'
            loader_epoch = logger.log_every(loader_train, header=header)
            
            module.train()
            if train_partitioner and self.partitioner is not None:
                self.partitioner.train()
            elif self.partitioner is not None:
                self.partitioner.eval()

            for i, data in enumerate(loader_epoch):
                _input, _label = data
                _label = _label.to(self.device)
                batch_size = int(_label.size(0))
                poison_num = int(batch_size * self.poison_percent)
                benign_num = batch_size - poison_num

                # BALANCED SAMPLING: Select poison samples to ensure balanced partition representation
                # This fixes the root cause of pseudo-label imbalance
                # Single pretrain-mode flag used throughout the loop
                is_pretrain_only = all([lf.__name__ == 'loss1' for lf in current_losses])

                if not is_pretrain_only and hasattr(self, 'class_partition_mapping'):
                    # Group batch indices by their target partition
                    partition_indices = {p: [] for p in range(self.nmarks)}
                    for idx in range(batch_size):
                        label_val = int(_label[idx].item())
                        target_partition = self.class_partition_mapping.get(label_val, label_val % self.nmarks)
                        partition_indices[target_partition].append(idx)
                    
                    # Select balanced subset: target samples_per_partition from each partition
                    balanced_poison_indices = []
                    samples_per_partition = max(1, poison_num // self.nmarks)  # ~2-3 for 9 poison / 4 partitions
                    
                    for partition_id in range(self.nmarks):
                        available = partition_indices[partition_id]
                        if available:
                            num_to_sample = min(samples_per_partition, len(available))
                            sampled = torch.tensor(available, dtype=torch.long, device=_label.device)[
                                torch.randperm(len(available), device=_label.device)[:num_to_sample]
                            ].tolist()
                            balanced_poison_indices.extend(sampled)
                    
                    # If we don't have enough samples yet, fill remaining slots from underrepresented partitions
                    remaining_slots = poison_num - len(balanced_poison_indices)
                    if remaining_slots > 0:
                        # Try to get more from partitions with available samples
                        for partition_id in range(self.nmarks):
                            if remaining_slots <= 0:
                                break
                            available = [idx for idx in partition_indices[partition_id] 
                                       if idx not in balanced_poison_indices]
                            if available:
                                num_to_add = min(remaining_slots, len(available))
                                additional = torch.tensor(available, dtype=torch.long, device=_label.device)[
                                    torch.randperm(len(available), device=_label.device)[:num_to_add]
                                ].tolist()
                                balanced_poison_indices.extend(additional)
                                remaining_slots -= num_to_add
                    
                    # Sort indices to maintain some order
                    balanced_poison_indices = sorted(balanced_poison_indices[:poison_num])
                    poison_indices_tensor = torch.tensor(balanced_poison_indices, dtype=torch.long, device=_input.device)
                    
                    # Get remaining indices for benign samples
                    all_indices = set(range(batch_size))
                    benign_indices_set = all_indices - set(balanced_poison_indices)
                    benign_indices_list = sorted(list(benign_indices_set))
                    benign_indices_tensor = torch.tensor(benign_indices_list, dtype=torch.long, device=_input.device)
                    
                    # Select poison and benign samples using indices
                    poisoned_input_clean = _input[poison_indices_tensor]
                    benign_input = _input[benign_indices_tensor].to(self.device)
                    poison_label = _label[poison_indices_tensor]
                    benign_label = _label[benign_indices_tensor]
                    
                    # Update counts in case we didn't get exactly poison_num
                    poison_num = len(poison_indices_tensor)
                    benign_num = len(benign_indices_tensor)
                else:
                    # Fallback to original simple selection (for pretrain or if no mapping)
                    poisoned_input_clean = _input[:poison_num, ...]
                    benign_input = _input[poison_num:, ...].to(self.device)
                    poison_label = _label[:poison_num]
                    benign_label = _label[poison_num:]

                # DEBUG: Print batch info for first few batches
                if DEBUG_VERBOSE and i < 3:  # Only debug first 3 batches
                    print(f"\n--- BATCH {i+1} DEBUG ---")
                    print(f"Batch size: {batch_size}, Poison num: {poison_num}, Benign num: {benign_num}")
                    print(f"Input shape: {_input.shape}, Label shape: {_label.shape}")
                    print(f"Label range: {_label.min().item()} to {_label.max().item()}")
                
                if DEBUG_VERBOSE and i < 3:  # Only debug first 3 batches
                    print(f"Poisoned input shape: {poisoned_input_clean.shape}")
                    print(f"Benign input shape: {benign_input.shape}")
                    print(f"Poison labels: {poison_label.cpu().numpy()}")
                    
                    # Show partition distribution of poison samples
                    if hasattr(self, 'class_partition_mapping'):
                        poison_partitions = [self.class_partition_mapping.get(int(lbl.item()), int(lbl.item()) % self.nmarks) 
                                           for lbl in poison_label]
                        partition_counts = [poison_partitions.count(p) for p in range(self.nmarks)]
                        print(f"Poison sample partition distribution: {partition_counts} (partitions 0-{self.nmarks-1})")
                    
                    print(f"Benign labels: {benign_label.cpu().numpy()}")

                # 1. Benign Forward Pass (for main model accuracy)
                benign_output = module(benign_input)
                loss_benign = loss_fn_ce(benign_output, benign_label)
                
                if DEBUG_VERBOSE and i < 3:  # Only debug first 3 batches
                    print(f"Benign output shape: {benign_output.shape}")
                    print(f"Benign output range: {benign_output.min().item():.4f} to {benign_output.max().item():.4f}")
                    print(f"Benign loss: {loss_benign.item():.6f}")

                # 2. Original Projan Poison Loss Calculation
                mod_inputs = [self.add_mark(poisoned_input_clean, index=j).to(self.device) for j in range(self.nmarks)]
                _input_poison_all = torch.cat([benign_input] + mod_inputs) # Concat for single forward pass
                _output_poison_all = module(_input_poison_all)
                _output_benign, mod_outputs = _output_poison_all[:benign_num], [_output_poison_all[benign_num+(j*poison_num):benign_num+((j+1)*poison_num)] for j in range(self.nmarks)]
                
                if DEBUG_VERBOSE and i < 3:  # Only debug first 3 batches
                    print(f"Poison all input shape: {_input_poison_all.shape}")
                    print(f"Poison all output shape: {_output_poison_all.shape}")
                    print(f"Number of mod outputs: {len(mod_outputs)}")
                    for j, mod_out in enumerate(mod_outputs):
                        print(f"  Mod output {j} shape: {mod_out.shape}, range: {mod_out.min().item():.4f} to {mod_out.max().item():.4f}")
                
                # If this train() invocation is a pretrain (only benign loss), skip poison/partitioner/stateful computations

                poisoned_losses = torch.zeros((nloss), device=self.device)
                poison_loss_components = []
                if not is_pretrain_only:
                    for j, loss_fn in enumerate(current_losses):
                        # The "poisoned_losses" should only include poison-related losses.
                        # Benign loss (loss1) is calculated separately.
                        if loss_fn.__name__ != 'loss1':
                            component_loss = loss_fn(None, mod_outputs, poison_label, target, self.probs)
                            poisoned_losses[j] = component_loss
                            poison_loss_components.append(component_loss.detach())
                            if DEBUG_VERBOSE and i < 3:  # Only debug first 3 batches
                                print(f"  Loss {loss_fn.__name__}: {component_loss.item():.6f}")
                
                # We only sum the weights for the poison losses
                weighted_poison_losses = loss_weights * poisoned_losses
                loss_projan_poison = weighted_poison_losses.sum()

                if poison_loss_components:
                    poison_component_tensor = torch.stack(poison_loss_components)
                    poison_loss_min = float(poison_component_tensor.min().item())
                    poison_loss_max = float(poison_component_tensor.max().item())
                    poison_loss_mean = float(poison_component_tensor.mean().item())
                else:
                    poison_loss_min = 0.0
                    poison_loss_max = 0.0
                    poison_loss_mean = 0.0
                weighted_poison_sum = float(loss_projan_poison.item()) if isinstance(loss_projan_poison, torch.Tensor) else float(loss_projan_poison)
                poison_loss_negative_flag = float(weighted_poison_sum < 0.0)
                
                if DEBUG_VERBOSE and i < 3:  # Only debug first 3 batches
                    print(f"Poisoned losses: {poisoned_losses.detach().cpu().numpy()}")
                    print(f"Loss weights: {loss_weights.cpu().numpy()}")
                    print(f"Projan poison loss: {loss_projan_poison.item():.6f}")
                    print(f"Poison loss stats -> min: {poison_loss_min:.6f}, max: {poison_loss_max:.6f}, mean: {poison_loss_mean:.6f}")

                # Throttle negative poison-loss warnings to avoid log spam
                #if poison_loss_negative_flag and (i % 20 == 0):
                #    print(f"WARNING: Negative loss_projan_poison detected ({loss_projan_poison.item():.6f}).")

                # 3. Stateful Projan Loss Calculation
                # If this is a pretrain-only run, skip partitioner/stateful computations entirely.
                if is_pretrain_only:
                    # During pure pretrain we only minimize the benign classification loss.
                    loss = loss_benign
                    # Logging placeholders for metrics that would otherwise be computed
                    loss_projan_poison = torch.tensor(0.0, device=self.device)
                    loss_partition_ce_steady = torch.tensor(0.0, device=self.device)
                    loss_stateful_poison_normalized = torch.tensor(0.0, device=self.device)
                    # Backprop and optimizer steps for pretraining
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # Log scalar values and continue to next batch
                    acc1, acc5 = accuracy(benign_output, benign_label, num_classes=num_classes, topk=(1, 5))
                    logger.update(loss_total=loss.item())
                    logger.update(loss_benign=loss_benign.item())
                    logger.update(loss_projan_poison=float(loss_projan_poison))
                    logger.update(loss_partition=float(loss_partition_ce_steady))
                    logger.update(loss_stateful_poison=float(loss_stateful_poison_normalized))
                    logger.update(top1_acc=acc1)
                    logger.update(top5_acc=acc5)

                    # TensorBoard Logging for pretrain step
                    global_step = (_epoch - 1) * len_loader_train + i
                    self.writer.add_scalar('Loss/Total', loss.item(), global_step)
                    self.writer.add_scalar('Loss/Benign', loss_benign.item(), global_step)
                    self.writer.add_scalar('Loss/Projan_Poison', float(loss_projan_poison), global_step)
                    self.writer.add_scalar('Loss/Partition', float(loss_partition_ce_steady), global_step)
                    self.writer.add_scalar('Loss/Stateful_Poison', float(loss_stateful_poison_normalized), global_step)
                    self.writer.add_scalar('Debug/Lambda/Partition_Effective', 0.0, global_step)
                    self.writer.add_scalar('Debug/Lambda/Stateful_Effective', 0.0, global_step)
                    empty_cache()
                    # Skip the rest of the complex poison logic for pretrain
                    continue

                poisoned_features = self._extract_features(poisoned_input_clean.to(self.device))
                # Optional feature noise regularization for partitioner
                if train_partitioner and float(getattr(self, 'partitioner_feature_noise_std', 0.0)) > 0.0 and self.partitioner is not None:
                    with torch.no_grad():
                        # Compute per-feature std across the batch; fallback to global std if degenerate
                        feat_std = poisoned_features.detach().std(dim=0, keepdim=True)
                        if torch.isnan(feat_std).any() or (feat_std <= 0).all():
                            feat_std = poisoned_features.detach().std().clamp_min(1e-6).view(1, 1)
                        noise_scale = float(self.partitioner_feature_noise_std)
                        noise = torch.randn_like(poisoned_features) * (feat_std * noise_scale)
                    poisoned_features = poisoned_features + noise
                
                if DEBUG_VERBOSE and i < 3:  # Only debug first 3 batches
                    print(f"Poisoned features shape: {poisoned_features.shape}")
                    print(f"Poisoned features range: {poisoned_features.min().item():.4f} to {poisoned_features.max().item():.4f}")
                    print(f"Features require grad: {poisoned_features.requires_grad}")
                
                partitioner_logits = self.partitioner(poisoned_features)
                
                if DEBUG_VERBOSE and i < 3:  # Only debug first 3 batches
                    print(f"Partitioner logits shape: {partitioner_logits.shape}")
                    print(f"Partitioner logits range: {partitioner_logits.min().item():.4f} to {partitioner_logits.max().item():.4f}")
                
                # 3a. Partition Loss (L_partition)
                if poison_num > 0:
                    cpu_labels = poison_label.detach().cpu().tolist()
                    partition_indices = []
                    for lbl in cpu_labels:
                        lbl_int = int(lbl)
                        if self.class_to_partition:
                            partition_indices.append(self.class_to_partition.get(lbl_int, lbl_int % self.nmarks))
                        else:
                            partition_indices.append(lbl_int % self.nmarks)
                    partition_pseudo_labels = torch.tensor(partition_indices, device=self.device, dtype=torch.long)
                    # Apply label smoothing if supported (PyTorch >= 1.10); otherwise fallback gracefully
                    try:
                        loss_partition_ce = F.cross_entropy(
                            partitioner_logits,
                            partition_pseudo_labels,
                            label_smoothing=float(getattr(self, 'partitioner_label_smoothing', 0.0))
                        )
                    except TypeError:
                        loss_partition_ce = F.cross_entropy(partitioner_logits, partition_pseudo_labels)
                    avg_prob = torch.softmax(partitioner_logits, dim=1).mean(dim=0)
                else:
                    partition_pseudo_labels = torch.zeros((0,), device=self.device, dtype=torch.long)
                    loss_partition_ce = torch.tensor(0.0, device=self.device)
                    avg_prob = torch.zeros(self.nmarks, device=self.device)
                
                if DEBUG_VERBOSE and i < 3:  # Only debug first 3 batches
                    print(f"Partition pseudo labels: {partition_pseudo_labels.cpu().numpy()}")
                    # Show distribution of pseudo-labels in this batch
                    if len(partition_pseudo_labels) > 0:
                        pseudo_hist = torch.bincount(partition_pseudo_labels, minlength=self.nmarks).float()
                        pseudo_dist = pseudo_hist / pseudo_hist.sum()
                        print(f"Pseudo-label distribution: {pseudo_dist.cpu().numpy()}")
                    print(f"Avg partition probs: {avg_prob.detach().cpu().numpy()}")
                    print(f"Partition loss (CE): {loss_partition_ce.item():.6f}")

                # 3b. Stateful Projan Loss (L_stateful_poison)
                loss_stateful_poison = torch.tensor(0.0, device=self.device)
                if stateful_enabled and poison_num > 0 and self.partitioner is not None:
                    predicted_partitions = partitioner_logits.argmax(dim=1)
                    y_target = torch.full_like(poison_label, target)

                    if DEBUG_VERBOSE and i < 3:  # Only debug first 3 batches
                        print(f"Predicted partitions: {predicted_partitions.cpu().numpy()}")
                        print(f"Target labels: {y_target.cpu().numpy()}")

                    for j in range(self.nmarks):
                        mask = (predicted_partitions == j)
                        if mask.sum() == 0:
                            if DEBUG_VERBOSE and i < 3:  # Debug block
                                print(f"  Partition {j}: no samples (mask sum = 0)")
                            continue

                        ms = float(mask.sum().item())
                        if DEBUG_VERBOSE and i < 3:  # Debug block
                            print(f"  Partition {j}: {ms} samples")

                        # Success Term: Trigger 'j' should work on inputs in partition 'j'
                        success_loss = loss_fn_ce(mod_outputs[j][mask], y_target[mask])
                        # Failure Term: Other triggers 'k' should NOT work on inputs in partition 'j'
                        failure_loss_sum = 0.0
                        for k in range(self.nmarks):
                            if k == j:
                                continue
                            failure_loss = loss_fn_ce(mod_outputs[k][mask], poison_label[mask])
                            failure_loss_sum = failure_loss_sum + failure_loss
                            if DEBUG_VERBOSE and i < 3:  # Debug block
                                print(f"    Failure loss for trigger {k} on partition {j}: {failure_loss.item():.6f}")

                        # Normalize by number of samples in this partition so each partition contributes equally
                        per_partition_loss = (success_loss + failure_loss_sum) / (ms + 1e-8)
                        loss_stateful_poison = loss_stateful_poison + per_partition_loss

                        if DEBUG_VERBOSE and i < 3:  # Debug block

                            print(f"    Success loss for partition {j}: {success_loss.item():.6f}")
                            print(f"    Per-partition normalized contribution: {per_partition_loss.item():.6f}")

                    # Keep the full aggregate penalty instead of normalizing by partition count
                    loss_stateful_poison_normalized = loss_stateful_poison
                    if DEBUG_VERBOSE and i < 3:
                        total_stateful_value = float(loss_stateful_poison.item()) if isinstance(loss_stateful_poison, torch.Tensor) else float(loss_stateful_poison)
                        normalized_stateful_value = float(loss_stateful_poison_normalized.item()) if isinstance(loss_stateful_poison_normalized, torch.Tensor) else float(loss_stateful_poison_normalized)
                        print(f"Total (summed) stateful poison loss: {total_stateful_value:.6f}")
                        print(f"Normalized stateful poison loss (avg over partitions): {normalized_stateful_value:.6f} (divided by {self.nmarks})")
                else:
                    if stateful_enabled and i < 3:
                        print("Stateful loss skipped (no poisoned samples in batch or partitioner unavailable).")
                    loss_stateful_poison_normalized = torch.tensor(0.0, device=self.device)

                # 4. Combine All Losses
                if is_pretrain_only:
                    lambda_partition_eff = 0.0
                    lambda_stateful_eff = 0.0
                else:
                    lambda_partition_eff = self.lambda_partition
                    lambda_stateful_eff = self.lambda_stateful
                loss = (loss_benign * (1-self.poison_percent) +
                        loss_projan_poison * self.poison_percent + 
                        lambda_partition_eff * loss_partition_ce +
                        lambda_stateful_eff * loss_stateful_poison_normalized)
                
                if DEBUG_VERBOSE and i < 3:  # Only debug first 3 batches
                    print(f"\n--- LOSS BREAKDOWN ---")
                    print(f"Loss benign: {loss_benign.item():.6f} * {1-self.poison_percent:.3f} = {loss_benign.item() * (1-self.poison_percent):.6f}")
                    print(f"Loss projan poison: {loss_projan_poison.item():.6f} * {self.poison_percent:.3f} = {loss_projan_poison.item() * self.poison_percent:.6f}")
                    print(f"Loss partition (stateful add-on): {loss_partition_ce.item():.6f} * {lambda_partition_eff:.3f} = {lambda_partition_eff * loss_partition_ce.item():.6f}")
                    loss_stateful_scalar = float(loss_stateful_poison_normalized.item()) if isinstance(loss_stateful_poison_normalized, torch.Tensor) else float(loss_stateful_poison_normalized)
                    print(f"Loss stateful (stateful add-on): {loss_stateful_scalar:.6f} * {lambda_stateful_eff:.3f} = {lambda_stateful_eff * loss_stateful_scalar:.6f}")
                    print(f"TOTAL LOSS: {loss.item():.6f}")

                # 5. Backward Pass and Optimizer Steps
                optimizer.zero_grad()
                if optimizer_partitioner is not None and train_partitioner:
                    optimizer_partitioner.zero_grad()
                
                # DEBUG: Check gradients before backward pass
                if DEBUG_VERBOSE and i < 3:  # Only debug first 3 batches
                    print(f"\n--- GRADIENT DEBUG ---")
                    # Check if any parameters have gradients before backward pass
                    has_grad_before = any(p.grad is not None for p in module.parameters())
                    print(f"Model has gradients before backward: {has_grad_before}")
                    
                    if hasattr(self, 'partitioner') and self.partitioner is not None:
                        has_grad_part_before = any(p.grad is not None for p in self.partitioner.parameters())
                        print(f"Partitioner has gradients before backward: {has_grad_part_before}")
                
                loss.backward()
                
                # DEBUG: Check gradients after backward pass
                if DEBUG_VERBOSE and i < 3:  # Only debug first 3 batches
                    has_grad_after = any(p.grad is not None for p in module.parameters())
                    print(f"Model has gradients after backward: {has_grad_after}")
                    
                    if has_grad_after:
                        # Check gradient norms
                        total_grad_norm = 0
                        for p in module.parameters():
                            if p.grad is not None:
                                total_grad_norm += p.grad.data.norm(2).item() ** 2
                        total_grad_norm = total_grad_norm ** 0.5
                        print(f"Model total gradient norm: {total_grad_norm:.6f}")
                    
                    if hasattr(self, 'partitioner') and self.partitioner is not None:
                        has_grad_part_after = any(p.grad is not None for p in self.partitioner.parameters())
                        print(f"Partitioner has gradients after backward: {has_grad_part_after}")
                        
                        if has_grad_part_after:
                            # Check partitioner gradient norms
                            total_grad_norm_part = 0
                            for p in self.partitioner.parameters():
                                if p.grad is not None:
                                    total_grad_norm_part += p.grad.data.norm(2).item() ** 2
                            total_grad_norm_part = total_grad_norm_part ** 0.5
                            print(f"Partitioner total gradient norm: {total_grad_norm_part:.6f}")
                
                optimizer.step()
                
                # Also step the partitioner optimizer if it exists
                if optimizer_partitioner is not None and train_partitioner:
                    # Optional gradient clipping to improve generalization stability
                    max_norm = float(getattr(self, 'partitioner_grad_clip', 0.0))
                    if max_norm and max_norm > 0.0:
                        try:
                            torch.nn.utils.clip_grad_norm_(self.partitioner.parameters(), max_norm)
                        except Exception as _clip_exc:
                            print(f"Warning: partitioner grad clipping failed: {_clip_exc}")
                    optimizer_partitioner.step()
                    if DEBUG_VERBOSE and i < 3:  # Only debug first 3 batches
                        print(f"Partitioner optimizer step completed")
                
                if DEBUG_VERBOSE and i < 3:  # Only debug first 3 batches
                    print(f"Optimizer step completed")
                    print(f"--- END BATCH {i+1} DEBUG ---\n")

                # Logging
                acc1, acc5 = accuracy(benign_output, benign_label, num_classes=num_classes, topk=(1, 5))
                logger.update(loss_total=loss.item())
                logger.update(loss_benign=loss_benign.item())
                logger.update(loss_projan_poison=loss_projan_poison.item())
                logger.update(loss_partition=loss_partition_ce.item())
                logger.update(loss_stateful_poison=loss_stateful_poison_normalized.item() if isinstance(loss_stateful_poison_normalized, torch.Tensor) else loss_stateful_poison_normalized)
                logger.update(top1_acc=acc1)
                logger.update(top5_acc=acc5)
                
                # TensorBoard Logging
                global_step = (_epoch - 1) * len_loader_train + i
                self.writer.add_scalar('Loss/Total', loss.item(), global_step)
                self.writer.add_scalar('Loss/Benign', loss_benign.item(), global_step)
                self.writer.add_scalar('Loss/Projan_Poison', loss_projan_poison.item(), global_step)
                self.writer.add_scalar('Loss/Partition', loss_partition_ce.item(), global_step)
                self.writer.add_scalar('Loss/Stateful_Poison', 
                                      loss_stateful_poison_normalized.item() if isinstance(loss_stateful_poison_normalized, torch.Tensor) else loss_stateful_poison_normalized, 
                                      global_step)
                # Log effective lambdas
                self.writer.add_scalar('Debug/Lambda/Partition_Effective', lambda_partition_eff, global_step)
                self.writer.add_scalar('Debug/Lambda/Stateful_Effective', lambda_stateful_eff, global_step)
                self.writer.add_scalar('Debug/ProjanPoison/Min', poison_loss_min, global_step)
                self.writer.add_scalar('Debug/ProjanPoison/Max', poison_loss_max, global_step)
                self.writer.add_scalar('Debug/ProjanPoison/Mean', poison_loss_mean, global_step)
                self.writer.add_scalar('Debug/ProjanPoison/WeightedSum', weighted_poison_sum, global_step)
                self.writer.add_scalar('Debug/ProjanPoison/NegativeFlag', poison_loss_negative_flag, global_step)
                self.writer.add_scalar('Accuracy/Top1_Benign', acc1, global_step)
                self.writer.add_scalar('Accuracy/Top5_Benign', acc5, global_step)
                
                # Log learning rate
                if optimizer.param_groups:
                    self.writer.add_scalar('Learning_Rate/Model', optimizer.param_groups[0]['lr'], global_step)
                if optimizer_partitioner and optimizer_partitioner.param_groups:
                    self.writer.add_scalar('Learning_Rate/Partitioner', optimizer_partitioner.param_groups[0]['lr'], global_step)
                
                empty_cache()

            if lr_scheduler:
                lr_scheduler.step()

            if validate_interval != 0 and (_epoch % validate_interval == 0 or _epoch == epoch):
                print(f'Epoch [{_epoch}/{epoch}] Training Stats: {str(logger)}')
                
                # Log partition distribution to detect collapse
                if hasattr(self, 'partitioner') and self.partitioner is not None:
                    with torch.no_grad():
                        sample_features = []
                        sample_targets = []
                        for data in loader_train:
                            _input, _label_sample = data
                            _input = _input.to(self.device)
                            feats = self._extract_features(_input[:min(32, len(_input))])
                            sample_features.append(feats)
                            if self.class_to_partition:
                                mapped_targets = [
                                    self.class_to_partition.get(int(lbl), int(lbl) % self.nmarks)
                                    for lbl in _label_sample[:min(32, len(_label_sample))].detach().cpu().tolist()
                                ]
                            else:
                                mapped_targets = [
                                    int(lbl) % self.nmarks
                                    for lbl in _label_sample[:min(32, len(_label_sample))].detach().cpu().tolist()
                                ]
                            sample_targets.append(torch.tensor(mapped_targets, device=self.device, dtype=torch.long))
                            if len(sample_features) >= 10:
                                break
                        if sample_features and sample_targets:
                            sample_features = torch.cat(sample_features, dim=0)
                            sample_targets = torch.cat(sample_targets, dim=0)
                            sample_logits = self.partitioner(sample_features)
                            sample_preds = sample_logits.argmax(dim=1)
                            partition_dist = torch.bincount(sample_preds, minlength=self.nmarks).float()
                            partition_dist = partition_dist / partition_dist.sum()
                            train_partition_acc = (sample_preds == sample_targets[:sample_preds.size(0)]).float().mean().item()
                            print(f'Partition distribution: {partition_dist.cpu().numpy()}')
                            print(f'Partitioner mapping accuracy (train sample): {train_partition_acc * 100:.2f}%')
                            if hasattr(self, 'writer') and self.writer is not None:
                                epoch_step = _epoch * max(1, len_loader_train)
                                for idx, freq in enumerate(partition_dist.cpu().tolist()):
                                    self.writer.add_scalar(f'Training/Partition_Predicted_{idx}', freq, epoch_step)
                                self.writer.add_scalar('Training/Partitioner_Accuracy_Sample', train_partition_acc, epoch_step)
                
                print(f'Epoch [{_epoch}/{epoch}] Results on the validation set: ==========')
                _, cur_acc, target_accs_list = validate_fn(module=module, num_classes=num_classes, loader=loader_valid, 
                                            get_data_fn=self.get_data, loss_fn=loss_fn_ce, epoch=_epoch, **kwargs)
                if cur_acc > best_acc:
                    best_acc = cur_acc
                    if save:
                        self.save(verbose=True)
        module.zero_grad()
        if save:
            print("\n--- Training complete. Saving final model. ---")
            self.save(suffix='final_epoch', verbose=True)
            
            # Save final metrics to JSON
            final_metrics = {
                'clean_accuracy': float(best_acc),
                'asr_overall': float(cur_acc) if 'cur_acc' in dir() else 0.0,
                'asr_per_trigger': [float(acc) for acc in target_accs_list] if 'target_accs_list' in dir() else [],
                'epoch': epoch
            }
            metrics_path = os.path.join(self.folder_path, 'final_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(final_metrics, f, indent=4)
            print(f"Final metrics saved to {metrics_path}")
    def validate_fn(self,
                    get_data_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] = None,
                    loss_fn: Callable[..., torch.Tensor] = None,
                    main_tag: str = 'valid', indent: int = 0, epoch: int = None, **kwargs) -> tuple[float, float]:
        #note!! in the following call, get_data_fn is None, so the get_data of the model is called, not the attack.
        _, clean_acc = self.model._validate(print_prefix='Validate Clean', main_tag='valid clean',
                                            get_data_fn=None, indent=indent, **kwargs)

        target_accs = [0] * self.nmarks
        individual_asrs = []  # Store individual ASRs for TensorBoard
        corrects1 = [None] * self.nmarks
        correct = npa(False)
        for j in range(self.nmarks):
            # poison_label and 'which' and get_data are sent to the model._validate function. This function, in turn,
            # calls get_data with poison_label and 'which'.
            _, target_accs[j] = self.model._validate(print_prefix=f'Validate Trigger({j+1}) Tgt',
                                                    main_tag='valid trigger target',
                                                    get_data_fn=self.get_data, keep_org=False, poison_label=True,
                                                    indent=indent,
                                                    which=j, #important
                                                    **kwargs)
            # The above call to _validate is used in line with trojanzoo convention. But it don't provide
            # instance-level details. So, we call correctness() to combine the results.
            corrects1[j] = self.correctness(print_prefix='Validate Trigger Tgt', main_tag='valid trigger target',
                                        keep_org=False, poison_label=True, which=j, **kwargs)
            
            # Calculate and print individual ASR for this trigger
            asr_j = 100 * corrects1[j].sum() / len(corrects1[j]) if len(corrects1[j]) > 0 else 0.0
            individual_asrs.append(asr_j)
            print(f'ASR for Trigger {j+1}: {asr_j:.2f}%')
            
            correct = np.logical_or(correct, corrects1[j])

        target_acc = 100*correct.sum()/len(correct)
        print('OR of [Trigger Tgt] on all triggers: ', 100 * correct.sum() / len(correct))

        corrects2 = [None]*self.nmarks
        correct = np.zeros((0,))
        for j in range(self.nmarks):
            self.model._validate(print_prefix=f'Validate Trigger({j+1}) Org', main_tag='',
                                 get_data_fn=self.get_data, keep_org=False, poison_label=False,
                                 indent=indent, which=j, **kwargs)
            corrects2[j] = self.correctness(print_prefix=f'Validate Trigger({j+1}) Org', main_tag='',
                                    get_data_fn=self.get_data, keep_org=False, poison_label=False,
                                    indent=indent, which=j, **kwargs)
            correct = np.concatenate((correct, corrects2[j]))


        print('average score of [Trigger Org] on all triggers: ', 100*correct.sum()/len(correct))
        #print(correct1.sum(), len(correct1), correct2.sum(), len(correct2), corrects.sum(), len(corrects))
        #print(100*correct2.sum()/len(correct2))

        # check the ASR when all triggers used together
        _, all_tgt_acc =  self.model._validate(print_prefix=f'Validate Combo Tgt',
                                                 main_tag='valid combo tgt',
                                                 get_data_fn=self.get_data, keep_org=False, poison_label=True,
                                                 indent=indent,
                                                 which=-1,  # important
                                                 **kwargs)

        # check the benign accuracy when all triggers used together
        _, all_clean_acc =  self.model._validate(print_prefix=f'Validate Combo Clean',
                                                 main_tag='valid combo org',
                                                 get_data_fn=self.get_data, keep_org=True, poison_label=False,
                                                 indent=indent,
                                                 which=None,
                                                 **kwargs)

        # Export a few attribution maps per epoch for visualization
        if self.xai_enable and epoch is not None:
            try:
                print(f"[XAI] Exporting attribution maps to {self.xai_dir}...")
                val_loader = self.dataset.get_dataloader('valid', batch_size=1, shuffle=False)
                saved = 0

                # Lazy import Captum to avoid hard dependency during training (use importlib to prevent static import issues)
                try:
                    import importlib
                    _captum_attr = importlib.import_module('captum.attr')
                    IntegratedGradients = getattr(_captum_attr, 'IntegratedGradients')
                    NoiseTunnel = getattr(_captum_attr, 'NoiseTunnel')
                    LayerGradCam = getattr(_captum_attr, 'LayerGradCam')
                    LayerAttribution = getattr(_captum_attr, 'LayerAttribution')
                except Exception as _captum_exc:
                    print(f"[XAI] Captum not available ({_captum_exc}). Please install 'captum' to enable IG/SmoothGrad/Grad-CAM.")
                    raise

                # Helper to resolve a feature layer module for Grad-CAM (tries common paths)
                def _resolve_feature_layer_module(model, layer_name: str):
                    candidates = [
                        getattr(model, layer_name, None),
                        getattr(getattr(model, 'model', None), layer_name, None),
                        getattr(getattr(model, 'backbone', None), layer_name, None),
                        getattr(getattr(model, 'features', None), layer_name, None),
                    ]
                    for mod in candidates:
                        if mod is not None:
                            return mod
                    return None

                os.makedirs(self.xai_dir, exist_ok=True)
                epoch_dir = os.path.join(self.xai_dir, f"epoch_{int(epoch):03d}")
                os.makedirs(epoch_dir, exist_ok=True)

                self.model.eval()
                device = self.device

                ig = IntegratedGradients(self.model)
                nt = NoiseTunnel(ig)
                feature_layer_module = _resolve_feature_layer_module(self.model, getattr(self, 'feature_layer', 'layer4'))
                layer_gradcam = None
                if feature_layer_module is not None:
                    try:
                        layer_gradcam = LayerGradCam(self.model, feature_layer_module)
                    except Exception as _lgc_exc:
                        print(f"[XAI] Warning: Failed to initialize LayerGradCam: {_lgc_exc}")

                # Limit total exported samples
                max_samples = max(1, int(self.xai_samples))

                sample_idx = 0
                for data in val_loader:
                    if saved >= max_samples:
                        break
                    x, y = data
                    x = x.to(device)
                    y = y.to(device)
                    # Use first item of the batch (batch_size=1 by default)
                    x = x[:1]
                    y = y[:1]

                    # Clean input target: predicted class for clean input
                    with torch.no_grad():
                        clean_logits = self.model(x)
                        clean_target_idx = int(clean_logits.argmax(dim=1).item())

                    # Prepare poisoned variants for each trigger until we reach the quota
                    for which in range(self.nmarks):
                        if saved >= max_samples:
                            break
                        x_poison = self.add_mark(x.clone(), which=which)

                        # Targets: clean uses its own prediction; poisoned uses backdoor target class
                        poison_target_idx = int(self.target_class)

                        # Ensure inputs require grad for attribution
                        x.requires_grad_(True)
                        x_poison.requires_grad_(True)

                        # Baseline for IG (zeros)
                        baseline = torch.zeros_like(x)

                        # 1) Integrated Gradients (clean and poison)
                        attr_clean_ig = ig.attribute(x, target=clean_target_idx, baselines=baseline, n_steps=32)
                        attr_poison_ig = ig.attribute(x_poison, target=poison_target_idx, baselines=baseline, n_steps=32)

                        # 2) SmoothGrad over IG
                        # Use squared smoothing to reduce noise; tune nt_samples/stdevs for runtime/quality tradeoff
                        attr_clean_sg = nt.attribute(
                            x,
                            target=clean_target_idx,
                            baselines=baseline,
                            nt_type='smoothgrad_sq',
                            nt_samples=20,
                            stdevs=0.1,
                        )
                        attr_poison_sg = nt.attribute(
                            x_poison,
                            target=poison_target_idx,
                            baselines=baseline,
                            nt_type='smoothgrad_sq',
                            nt_samples=20,
                            stdevs=0.1,
                        )

                        # 3) Grad-CAM (on chosen feature layer), if available
                        attr_clean_gc = None
                        attr_poison_gc = None
                        if layer_gradcam is not None:
                            try:
                                lgc_clean = layer_gradcam.attribute(x, target=clean_target_idx)
                                lgc_poison = layer_gradcam.attribute(x_poison, target=poison_target_idx)
                                # Upsample to input size and collapse channel dimension
                                lgc_clean_up = LayerAttribution.interpolate(lgc_clean, x.shape[-2:])
                                lgc_poison_up = LayerAttribution.interpolate(lgc_poison, x.shape[-2:])
                                attr_clean_gc = lgc_clean_up.sum(dim=1, keepdim=True)
                                attr_poison_gc = lgc_poison_up.sum(dim=1, keepdim=True)
                            except Exception as _gc_exc:
                                print(f"[XAI] Warning: Grad-CAM computation failed: {_gc_exc}")

                        # Convert attributions to single-channel heatmaps (absolute-sum over channels)
                        def _to_heatmap(t: torch.Tensor) -> torch.Tensor:
                            if t is None:
                                return None
                            if t.dim() == 4 and t.size(1) > 1:
                                t = t.abs().sum(dim=1, keepdim=True)
                            elif t.dim() == 4 and t.size(1) == 1:
                                t = t.abs()
                            # Min-max normalize per image
                            t_min = t.amin(dim=(2,3), keepdim=True)
                            t_max = t.amax(dim=(2,3), keepdim=True)
                            t = (t - t_min) / (t_max - t_min + 1e-8)
                            return t.clamp(0, 1).detach().cpu()

                        hm_clean_ig = _to_heatmap(attr_clean_ig)
                        hm_poison_ig = _to_heatmap(attr_poison_ig)
                        hm_clean_sg = _to_heatmap(attr_clean_sg)
                        hm_poison_sg = _to_heatmap(attr_poison_sg)
                        hm_clean_gc = _to_heatmap(attr_clean_gc) if attr_clean_gc is not None else None
                        hm_poison_gc = _to_heatmap(attr_poison_gc) if attr_poison_gc is not None else None

                        # Also normalize and save the original input for context
                        def _to_image(t: torch.Tensor) -> torch.Tensor:
                            # assumes input is in [0,1] or normalized; clip for saving
                            return t.detach().cpu().clamp(0, 1)

                        x_cpu = _to_image(x)
                        x_poison_cpu = _to_image(x_poison)

                        # Build filenames and save
                        base = f"sample{sample_idx:04d}_trig{which}"
                        # Inputs
                        save_image(x_cpu, os.path.join(epoch_dir, f"{base}_clean_input.png"))
                        save_image(x_poison_cpu, os.path.join(epoch_dir, f"{base}_poison_input.png"))
                        # IG
                        save_image(hm_clean_ig, os.path.join(epoch_dir, f"{base}_clean_ig.png"))
                        save_image(hm_poison_ig, os.path.join(epoch_dir, f"{base}_poison_ig.png"))
                        # SmoothGrad
                        save_image(hm_clean_sg, os.path.join(epoch_dir, f"{base}_clean_smoothgrad.png"))
                        save_image(hm_poison_sg, os.path.join(epoch_dir, f"{base}_poison_smoothgrad.png"))
                        # Grad-CAM if available
                        if hm_clean_gc is not None and hm_poison_gc is not None:
                            save_image(hm_clean_gc, os.path.join(epoch_dir, f"{base}_clean_gradcam.png"))
                            save_image(hm_poison_gc, os.path.join(epoch_dir, f"{base}_poison_gradcam.png"))

                        saved += 1
                        sample_idx += 1

                print(f"[XAI] Saved {saved} attribution sample(s) for epoch {epoch}.")
            except Exception as exc:
                print(f"[XAI] ERROR: Failed to export attribution maps: {exc}")
                import traceback
                traceback.print_exc()

        for j in range(self.nmarks):
            prints(f'Validate Confidence({j+1}): {self.validate_confidence(which=j):.3f}', indent=indent)
            prints(f'Neuron Jaccard Idx({j+1}): {self.check_neuron_jaccard(which=j):.3f}', indent=indent)

        partitioner_acc = None
        partition_hist = None
        partition_confusion = None
        target_hist_counts = None
        first_batch_pred_counts = None
        first_batch_target_counts = None
        stateful_success_hist = None
        stateful_samples_hist = None
        misfire_hist = None

        if getattr(self, 'partitioner', None) is not None and self.class_to_partition:
            partitioner_training = self.partitioner.training
            self.partitioner.eval()
            try:
                val_loader = self.dataset.get_dataloader('valid', batch_size=256, shuffle=False)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"WARNING: Unable to build validation loader for partitioner eval ({exc}).")
                val_loader = None

            if val_loader is not None:
                total = 0
                correct_partitions = 0
                partition_hist = torch.zeros(self.nmarks, dtype=torch.long)
                target_hist = torch.zeros(self.nmarks, dtype=torch.long)
                confusion = torch.zeros((self.nmarks, self.nmarks), dtype=torch.long)
                first_pred_hist = None
                first_target_hist = None
                stateful_batch_limit = 3
                stateful_batches_processed = 0
                misfire_counts = torch.zeros((self.nmarks, self.nmarks), dtype=torch.long)
                success_counts = torch.zeros(self.nmarks, dtype=torch.long)
                samples_per_part = torch.zeros(self.nmarks, dtype=torch.long)
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        features = self._extract_features(inputs)
                        logits = self.partitioner(features)
                        preds = logits.argmax(dim=1)

                        targets = self._labels_to_partitions(labels)

                        correct_partitions += (preds == targets).sum().item()
                        total += preds.size(0)
                        preds_cpu = preds.detach().cpu()
                        targets_cpu = targets.detach().cpu()
                        partition_hist += torch.bincount(preds_cpu, minlength=self.nmarks)
                        target_hist += torch.bincount(targets_cpu, minlength=self.nmarks)

                        batch_index = targets_cpu * self.nmarks + preds_cpu
                        batch_conf = torch.bincount(batch_index, minlength=self.nmarks * self.nmarks)
                        confusion += batch_conf.view(self.nmarks, self.nmarks)

                        if first_pred_hist is None:
                            first_pred_hist = torch.bincount(preds_cpu, minlength=self.nmarks)
                            first_target_hist = torch.bincount(targets_cpu, minlength=self.nmarks)

                        if stateful_batches_processed < stateful_batch_limit:
                            stateful_batches_processed += 1
                            mod_preds = []
                            for trig_idx in range(self.nmarks):
                                trig_input = self.add_mark(inputs, trig_idx)
                                trig_output = self.model(trig_input)
                                mod_preds.append(trig_output.argmax(dim=1))

                            for part_idx in range(self.nmarks):
                                part_mask = preds == part_idx
                                samples = int(part_mask.sum().item())
                                if samples == 0:
                                    continue
                                samples_per_part[part_idx] += samples
                                success_hits = int((mod_preds[part_idx][part_mask] == self.target_class).sum().item())
                                success_counts[part_idx] += success_hits
                                for trig_idx in range(self.nmarks):
                                    if trig_idx == part_idx:
                                        continue
                                    wrong_hits = int((mod_preds[trig_idx][part_mask] == self.target_class).sum().item())
                                    if wrong_hits:
                                        misfire_counts[part_idx, trig_idx] += wrong_hits

                if total > 0:
                    partitioner_acc = 100.0 * correct_partitions / total
                    print(f'Partitioner mapping accuracy: {partitioner_acc:.2f}% over {total} samples.')
                else:
                    print('WARNING: Partitioner validation loader produced zero samples.')

                if confusion.sum().item() > 0:
                    print('Partitioner confusion matrix (rows=target partitions, cols=predictions):')
                    for row_idx in range(self.nmarks):
                        row_total = int(confusion[row_idx].sum().item())
                        if row_total == 0:
                            print(f'  Target partition {row_idx}: total=0, distribution={[0.0]*self.nmarks}')
                            continue
                        row_dist = (confusion[row_idx].float() / row_total).cpu().numpy().round(3).tolist()
                        print(f'  Target partition {row_idx}: total={row_total}, distribution={row_dist}')

                if first_pred_hist is not None:
                    print(f'First batch predicted counts: {first_pred_hist.tolist()}')
                    print(f'First batch target counts:    {first_target_hist.tolist()}')

                if stateful_batches_processed > 0:
                    print('Stateful trigger coverage (first batches):')
                    for part_idx in range(self.nmarks):
                        total_samples = int(samples_per_part[part_idx].item())
                        if total_samples == 0:
                            print(f'  Partition {part_idx}: no samples observed.')
                            continue
                        success_hits = int(success_counts[part_idx].item())
                        success_rate = success_hits / max(total_samples, 1)
                        print(f'  Partition {part_idx}: samples={total_samples}, success_rate={success_rate:.3f}')
                        for trig_idx in range(self.nmarks):
                            if trig_idx == part_idx:
                                continue
                            wrong_hits = int(misfire_counts[part_idx, trig_idx].item())
                            if wrong_hits > 0:
                                print(f'    Trigger {trig_idx} misfired {wrong_hits} times')

                partition_confusion = confusion
                target_hist_counts = target_hist
                first_batch_pred_counts = first_pred_hist
                first_batch_target_counts = first_target_hist
                stateful_success_hist = success_counts
                stateful_samples_hist = samples_per_part
                misfire_hist = misfire_counts

            if partitioner_training:
                self.partitioner.train()

        if self.clean_acc - clean_acc > 30 and self.clean_acc > 40:  # TODO: better not hardcoded
            target_acc = 0.0
            for j in range(self.nmarks):
                target_accs[j] = 0.0
        
        # TensorBoard Logging for Validation Metrics
        if epoch is not None and hasattr(self, 'writer'):
            self.writer.add_scalar('Validation/Clean_Accuracy', clean_acc, epoch)
            self.writer.add_scalar('Validation/Overall_ASR', target_acc, epoch)
            self.writer.add_scalar('Validation/Combo_Target_Accuracy', all_tgt_acc, epoch)
            self.writer.add_scalar('Validation/Combo_Clean_Accuracy', all_clean_acc, epoch)
            
            # Log individual trigger ASRs
            for j, asr in enumerate(individual_asrs):
                self.writer.add_scalar(f'Validation/Trigger_{j+1}_ASR', asr, epoch)
            
            # Log confidence and Jaccard metrics
            for j in range(self.nmarks):
                conf = self.validate_confidence(which=j)
                jaccard = self.check_neuron_jaccard(which=j)
                self.writer.add_scalar(f'Validation/Trigger_{j+1}_Confidence', conf, epoch)
                self.writer.add_scalar(f'Validation/Trigger_{j+1}_Jaccard', jaccard, epoch)
            if partitioner_acc is not None:
                self.writer.add_scalar('Validation/Partitioner_Accuracy', partitioner_acc, epoch)
            if partition_hist is not None:
                total_predictions = max(partition_hist.sum().item(), 1)
                partition_hist = partition_hist.float() / float(total_predictions)
                for idx, freq in enumerate(partition_hist.tolist()):
                    self.writer.add_scalar(f'Validation/Partitioner_Predicted_{idx}', freq, epoch)
            if target_hist_counts is not None:
                total_targets = max(target_hist_counts.sum().item(), 1)
                target_ratio = target_hist_counts.float() / float(total_targets)
                for idx, freq in enumerate(target_ratio.tolist()):
                    self.writer.add_scalar(f'Validation/Partitioner_Target_{idx}', freq, epoch)
            if partition_confusion is not None:
                for idx in range(self.nmarks):
                    denom = max(partition_confusion[idx].sum().item(), 1)
                    row_acc = partition_confusion[idx, idx].float() / float(denom)
                    self.writer.add_scalar(f'Validation/Partitioner_RowAcc_{idx}', row_acc.item(), epoch)
            if first_batch_pred_counts is not None:
                for idx, count in enumerate(first_batch_pred_counts.tolist()):
                    self.writer.add_scalar(f'Validation/Partitioner_FirstBatchPred_{idx}', float(count), epoch)
            if first_batch_target_counts is not None:
                for idx, count in enumerate(first_batch_target_counts.tolist()):
                    self.writer.add_scalar(f'Validation/Partitioner_FirstBatchTarget_{idx}', float(count), epoch)
            if stateful_success_hist is not None and stateful_samples_hist is not None:
                for idx in range(self.nmarks):
                    total_samples = max(stateful_samples_hist[idx].item(), 1)
                    success_rate = stateful_success_hist[idx].float() / float(total_samples)
                    self.writer.add_scalar(f'Validation/Stateful_SuccessRate_{idx}', success_rate.item(), epoch)
            if misfire_hist is not None:
                for row in range(self.nmarks):
                    for col in range(self.nmarks):
                        if row == col:
                            continue
                        misfires = misfire_hist[row, col].item()
                        if misfires > 0:
                            self.writer.add_scalar(f'Validation/Stateful_Misfire_{row}_{col}', float(misfires), epoch)
            # EM pseudo-label metrics removed in the simplified configuration
        
        return clean_acc, target_acc, target_accs

    def correctness(self, keep_org=False, poison_label=True, which=0, **kwargs):
        if 'loader' in kwargs:
            loader = kwargs['loader']
        else:
            loader = self.dataset.loader['valid'] # todo: valid2
        self.model.eval()
        with torch.no_grad(): # todo does need to go inside loop?
            corrects = t = np.zeros((0,), dtype=bool)

            for data in loader:
                inp, label = self.get_data(data, mode='valid',
                                           keep_org=keep_org, poison_label=poison_label, which=which, **kwargs)
                inp = inp.to(env['device'])
                label = label.to(env['device'])
                output = self.model(inp)
                if torch.any(torch.isnan(output)):
                    print('warning: NaN in output.')
                pred = output.argmax(1)
                if label.ndim > 1:
                    label = label.argmax(1)
                #pred = pred.unsqueeze(0)
                correct = (pred == label).detach().cpu().numpy() #todo: handle the case of fractional labels.
                #correct = correct[0]
                corrects = np.concatenate((corrects, correct))
        return corrects

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor], keep_org: bool = True,
                 poison_label=True, which=None, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        #for now, keep_org is ignored and only used to stay consistent with trojanzoo.
        #todo handle keep_org
        x, y = data
        x = x.to(env['device'])
        y = y.to(env['device'])
        if poison_label:
            y[...] = self.target_class
            y = y.to(env['device'])
        if which is not None:
            if which >= 0:
                x = self.add_mark(x, which, **kwargs)
            else:  # use negative value to apply all triggers together
                for i in range(self.nmarks):
                    x = self.add_mark(x, which=i, **kwargs)
        return x, y


    def validate_confidence(self, which=0) -> float:
        confidence = SmoothedValue()
        with torch.no_grad():
            for data in self.dataset.loader['valid']:
                _input, _label = self.model.get_data(data)
                _input = _input.to(env['device'])
                _label = _label.to(env['device'])
                idx1 = _label != self.target_class
                _input = _input[idx1]
                _label = _label[idx1]
                if len(_input) == 0:
                    continue
                poison_input = self.add_mark(_input, which)
                poison_label = self.model.get_class(poison_input)
                idx2 = poison_label == self.target_class
                poison_input = poison_input[idx2]
                if len(poison_input) == 0:
                    continue
                batch_conf = self.model.get_prob(poison_input)[:, self.target_class].mean()
                confidence.update(batch_conf, len(poison_input))
        return confidence.global_avg

    def check_neuron_jaccard(self, ratio=0.5, which=0) -> float:
        feats_list = []
        poison_feats_list = []
        with torch.no_grad():
            for data in self.dataset.loader['valid']:
                _input, _label = self.model.get_data(data)
                _input = _input.to(env['device'])
                _label = _label.to(env['device'])
                poison_input = self.add_mark(_input, which)

                _feats = self.model.get_final_fm(_input)
                poison_feats = self.model.get_final_fm(poison_input)
                feats_list.append(_feats)
                poison_feats_list.append(poison_feats)
        feats_list = torch.cat(feats_list).mean(dim=0)
        poison_feats_list = torch.cat(poison_feats_list).mean(dim=0)
        length = int(len(feats_list) * ratio)
        _idx = set(feats_list.argsort(descending=True)[:length].tolist())
        poison_idx = set(poison_feats_list.argsort(descending=True)[:length].tolist())
        jaccard_idx = len(_idx & poison_idx) / len(_idx | poison_idx)
        return jaccard_idx
