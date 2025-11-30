import trojanvision.marks
from ..normal.badnet import BadNet
from trojanvision.marks import Watermark
from trojanzoo.environ import env
from trojanzoo.utils.memory import empty_cache

from trojanzoo.utils.output import prints
from trojanzoo.utils.logger import SmoothedValue
from trojanzoo.utils import to_list
from trojanzoo.utils.logger import MetricLogger, SmoothedValue
from trojanzoo.utils.model import accuracy, activate_params
from trojanzoo.utils.output import ansi, get_ansi_len, output_iter, prints
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from trojanzoo.utils.io import DictReader

import torch
import torch.nn as nn
from torch import tensor
import torch.nn.functional as F
import random
import numpy as np
from numpy import array as npa
import math
from typing import Callable
from tqdm import tqdm
import os
import argparse

from .losses import *



class Prob(BadNet):

    name: str = 'prob'

    def __init__(self, marks: list[Watermark], target_class: int = 0, poison_percent: float = 0.01,
                 train_mode: str = 'batch', probs: list[float] = None,
                 losses = ['loss1'],
                 init_loss_weights = None,
                 cbeta_epoch = -1,
                 disable_batch_norm = True,
                 batchnorm_momentum = None,
                 pretrain_epoch = 0,
                 **kwargs): #todo add cmd args
        super().__init__(marks[0], target_class, poison_percent, train_mode, **kwargs)
        self.marks: list[Watermark] = marks
        self.nmarks = len(self.marks)
        if probs is not None:
            # tolerate mismatched length: pad with last value or truncate with a warning
            if len(probs) != self.nmarks:
                prints(f"Warning: provided --probs length={len(probs)} != number of marks={self.nmarks}. "
                       "Padding (repeat last) or truncating to match.")
                if len(probs) < self.nmarks:
                    probs = list(probs) + [probs[-1]] * (self.nmarks - len(probs))
                else:
                    probs = list(probs)[:self.nmarks]
        else:
            probs = [1] * self.nmarks

        sump = sum(probs)
        # the following line is commented to allow for single trigger probabilistic tests
        # probs = [p/sump for p in probs]
        self.probs = probs
        # show the resolved probs when verbose
        try:
            if env.get('verbose', 0):
                prints(f"Resolved trigger probs: {self.probs}")
        except Exception:
            pass
        self.loss_names = losses
        self.losses = [get_loss_by_name(loss) for loss in losses]
        self.cbeta_epoch = cbeta_epoch
        self.init_loss_weights = npa(init_loss_weights) if init_loss_weights is not None else None
        if disable_batch_norm:
            self.model.disable_batch_norm()
        self.model.set_batchnorm_momentum(batchnorm_momentum)
        # note: the following fields are not updated when the model batchnorm is disabled/enabled/gets params changed.
        self.disable_batch_norm = disable_batch_norm
        self.batchnorm_momentum = batchnorm_momentum
        self.pretrain_epoch = pretrain_epoch
        # instrumentation for debugging loss contributions
        # accumulators: benign_sum (float), poisoned_sum (np.array), count (int)
        try:
            nloss_tmp = max(1, len(self.losses))
        except Exception:
            nloss_tmp = 1
        self._prob_stats = {
            'benign_sum': 0.0,
            'poisoned_sum': np.zeros((nloss_tmp,), dtype=float),
            'count': 0
        }
        # optional per-loss normalization warmup stats
        self.normalize_losses = bool(kwargs.get('normalize_losses', False))
        self.warmup_batches = int(kwargs.get('warmup_batches', 1000))
        # store one extra slot for the benign loss (index 0) followed by poisoned losses
        self._norm_stats = {
            'sum_abs': np.zeros((nloss_tmp + 1,), dtype=float),
            'norm_count': 0,
            'scales': None,
            'warmup_done': False
        }
        # used by the summary() method
        self.param_list['prob'] = ['probs', 'loss_names', 'cbeta_epoch', 'init_loss_weights',
                                   'disable_batch_norm', 'batchnorm_momentum', 'pretrain_epoch']
        # expose normalization params
        self.param_list['prob'] += ['normalize_losses', 'warmup_batches']


    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        super().add_argument(group)
        group.add_argument('--init_loss_weights', dest='init_loss_weights', type=float, nargs='*', default=None,
                           help='initial weights of losses which may be updated later')
        group.add_argument('--probs', dest='probs', type=float, nargs='*', default=None,
                           help='the expected success probability of attach, one entry per trigger')
        group.add_argument('--cbeta_epoch', dest='cbeta_epoch', type=int, default=-1) # todo: add help
        group.add_argument('--disable_batch_norm', dest='disable_batch_norm', type=bool, default=True,
                           help='disable batch normalization layers of the model')
        group.add_argument('--batchnorm_momentum', dest='batchnorm_momentum', type=float, default=None,
                           help='momentum hyper-parameter for batchnorm layers')
        group.add_argument('--pretrain_epoch', dest='pretrain_epoch', type=int, default=0,
                           help='number of epochs to pretrain network regularly before disabling batchnorm')
        group.add_argument('--losses', dest='losses', type=str, nargs='*', default=['loss1'],
                           help='names of loss functions')

        group.add_argument('--normalize_losses', dest='normalize_losses', action='store_true',
                help='enable running-mean normalization of poisoned losses (warmup)')
        group.add_argument('--warmup_batches', dest='warmup_batches', type=int, default=1000,
                help='number of batches to warmup and estimate mean abs(loss) before normalizing')

        type_map = {'mark_height': int, 'mark_width': int, 'height_offset': int, 'width_offset': int}
        group.add_argument('--extra_mark', action=DictReader, nargs='*', dest='extra_marks', type_map=type_map)

    def attack(self, epoch: int, save=False, **kwargs):
        # Delegate to the model training pipeline. Accept optimizer/lr_scheduler from kwargs
        # and pass a custom loss function that implements the probabilistic multi-trigger loss.
        loader_train = self.dataset.get_dataloader('train')
        loader_valid = self.dataset.get_dataloader('valid')

        optimizer = kwargs.get('optimizer', None)
        lr_scheduler = kwargs.get('lr_scheduler', None)

        # Stage 1: Pretrain with batch norm enabled, loss1 only (skip if pretrain_epoch <= 0)
        if self.pretrain_epoch and self.pretrain_epoch > 0:
            print(f"Pretrain stage: epochs={self.pretrain_epoch}, using losses: ['loss1']")
            self.model.enable_batch_norm()
            # use model's _train and pass a wrapper matching the train() expected signature.
            # train() calls loss_fn(_input, _label, _output=_output, amp=amp)
            # For Stage 1 pretraining, use clean labels (poison_label=False in get_data).
            def _loss1_adapter(_input, _label, _output=None, amp=False, **_k):
                # original loss1 signature: loss1(output, mod_outputs, label, target, probs)
                # In pretrain, _label is clean (not modified to target_class)
                return torch.nn.CrossEntropyLoss()(_output, _label)

            # Wrapper to ensure get_data passes poison_label=False in pretrain stage
            def _get_data_clean(data, mode=None, **_k):
                return self.get_data(data, mode=mode, poison_label=False, **_k)

            call_kwargs = dict(kwargs)
            # avoid passing optimizer/lr_scheduler twice (they may be present in kwargs)
            call_kwargs.pop('optimizer', None)
            call_kwargs.pop('lr_scheduler', None)
            self.model._train(epoch=self.pretrain_epoch, optimizer=optimizer, lr_scheduler=lr_scheduler,
                              save=save, loader_train=loader_train, loader_valid=loader_valid,
                              loss_fn=_loss1_adapter, validate_fn=self.validate_fn, get_data_fn=_get_data_clean,
                              save_fn=self.save, **call_kwargs)
        else:
            print("Pretrain stage skipped (pretrain_epoch <= 0)")

        # Stage 2: Full training with batch norm disabled using the probabilistic combined loss
        print(f"Full training stage starting: epochs={epoch}, using losses: {self.loss_names}")
        self.model.disable_batch_norm()
        call_kwargs = dict(kwargs)
        call_kwargs.pop('optimizer', None)
        call_kwargs.pop('lr_scheduler', None)
        self.model._train(epoch=epoch, optimizer=optimizer, lr_scheduler=lr_scheduler,
                          save=save, loader_train=loader_train, loader_valid=loader_valid,
                          loss_fn=self.prob_loss, validate_fn=self.validate_fn, get_data_fn=self.get_data,
                          save_fn=self.save, **call_kwargs)

    @staticmethod
    def oh_ce_loss(output, target):
        N = output.shape[0]
        output = F.log_softmax(output, 1)
        target = target.to(dtype=torch.float)
        output = torch.trace(-torch.matmul(output, target.transpose(1, 0))) / N
        return output

    def add_mark(self, x: torch.Tensor, index = 0, **kwargs) -> torch.Tensor:
        return self.marks[index].add_mark(x, **kwargs)

    def prob_loss(self, _input: torch.Tensor, _label: torch.Tensor, _output: torch.Tensor = None,
                  amp: bool = False, **kwargs) -> torch.Tensor:
        """
        Combined loss used by the model training loop. This computes benign loss on the clean
        portion of the batch and poisoned losses for poisoned portion using configured loss functions.
        """
        # ensure tensors on device
        device = env['device']
        batch_size = int(_label.size(0))
        poison_num = int(batch_size * self.poison_percent)
        nloss = len(self.losses)

        # prepare loss weights
        if self.init_loss_weights is not None and len(self.init_loss_weights) == nloss:
            loss_weights = torch.tensor(self.init_loss_weights, device=device, dtype=torch.float)
        else:
            loss_weights = torch.ones((nloss,), device=device, dtype=torch.float) / float(nloss)
        loss_weights = loss_weights / loss_weights.sum()

        # if no poisoned examples, use simple CE on whole batch
        if poison_num == 0:
            return torch.nn.CrossEntropyLoss()( _output, _label )

        poisoned_input = _input[:poison_num].to(device)
        benign_input = _input[poison_num:].to(device)
        benign_label = _label[poison_num:].to(device)

        # generate modified (triggered) inputs for each mark
        mod_inputs = [self.add_mark(poisoned_input, index=j).to(device) for j in range(self.nmarks)]
        mod_outputs = [self.model(mi) for mi in mod_inputs]

        poisoned_losses = torch.zeros((nloss,), device=device)
        for j, loss_fn in enumerate(self.losses):
            poisoned_losses[j] = loss_fn(_output[:poison_num, ...], mod_outputs, _label[:poison_num, ...],
                                         self.target_class, self.probs)
        # compute verbosity flag early for warmup prints
        try:
            verbose_flag = env.get('verbose', 0)
        except Exception:
            verbose_flag = 0

        # prepare numpy copy for accumulation/normalization
        try:
            poisoned_vals = poisoned_losses.detach().cpu().numpy().astype(float)
        except Exception:
            poisoned_vals = np.zeros((nloss,), dtype=float)

        # compute benign loss early so we can include it in normalization warmup
        # use loss_fns[0] for benign (matches original projan, which uses loss1)
        if len(benign_input) > 0:
            try:
                benign_loss = self.losses[0](_output[poison_num:, ...], None, benign_label, None, None)
            except Exception:
                benign_loss = torch.tensor(0.0, device=device)
        else:
            benign_loss = torch.tensor(0.0, device=device)

        # optional per-loss running-mean warmup: accumulate abs values (include benign as index 0)
        if self.normalize_losses:
            try:
                benign_abs = float(abs(benign_loss.detach().cpu().item())) if isinstance(benign_loss, torch.Tensor) else float(abs(benign_loss))
                poisoned_abs = np.abs(poisoned_vals)
                combined_abs = np.concatenate((np.array([benign_abs], dtype=float), poisoned_abs))
                # ensure shape
                if self._norm_stats['sum_abs'].shape[0] != combined_abs.shape[0]:
                    self._norm_stats['sum_abs'] = np.zeros_like(combined_abs)
                self._norm_stats['sum_abs'] += combined_abs
                self._norm_stats['norm_count'] += 1
                # check warmup completion
                if (not self._norm_stats['warmup_done']) and (self._norm_stats['norm_count'] >= self.warmup_batches):
                    mean_abs = self._norm_stats['sum_abs'] / float(self._norm_stats['norm_count'])
                    # avoid zeros
                    mean_abs[mean_abs == 0.0] = 1.0
                    self._norm_stats['scales'] = mean_abs
                    self._norm_stats['warmup_done'] = True
                    if verbose_flag:
                        prints(f"[prob_loss debug] normalization warmup complete; mean_abs={mean_abs}")
            except Exception:
                pass

        # apply normalization if warmup completed (scale benign and poisoned losses)
        if self.normalize_losses and self._norm_stats.get('warmup_done', False):
            try:
                scales = np.array(self._norm_stats['scales'], dtype=float)
                scales[scales == 0.0] = 1.0
                scales_t = torch.tensor(scales, device=device, dtype=torch.float)
                # first entry is benign scale
                benign_scale = scales_t[0]
                poisoned_scales_t = scales_t[1:]
                # scale losses
                try:
                    benign_loss = benign_loss / benign_scale
                except Exception:
                    pass
                try:
                    poisoned_losses = poisoned_losses / poisoned_scales_t
                except Exception:
                    pass
            except Exception:
                pass
        # --- instrumentation: accumulate per-batch benign & poisoned loss magnitudes ---
        # (benign_loss already computed above using loss_fns[0])

        L1 = loss_weights[0] * benign_loss * (1.0 - self.poison_percent)
        L2 = (loss_weights * poisoned_losses * self.poison_percent).sum()
        loss = L1 + L2

        try:
            # benign_loss is a tensor; convert to float
            benign_val = float(benign_loss.detach().cpu().item()) if isinstance(benign_loss, torch.Tensor) else float(benign_loss)
        except Exception:
            benign_val = float(0.0)

        # poisoned_vals contains the original (pre-normalization) per-loss magnitudes
        # ensure shape match
        if poisoned_vals.shape[0] != self._prob_stats['poisoned_sum'].shape[0]:
            self._prob_stats['poisoned_sum'] = np.zeros_like(poisoned_vals)

        # update accumulators
        self._prob_stats['benign_sum'] += benign_val
        self._prob_stats['poisoned_sum'] += poisoned_vals
        self._prob_stats['count'] += 1

        # print periodic diagnostics when verbose
        try:
            verbose_flag = env.get('verbose', 0)
        except Exception:
            verbose_flag = 0

        if verbose_flag and (self._prob_stats['count'] % 50 == 0):
            avg_benign = self._prob_stats['benign_sum'] / float(self._prob_stats['count'])
            avg_poison = self._prob_stats['poisoned_sum'] / float(self._prob_stats['count'])
            # human-readable string
            s = f"[prob_loss debug] batches={self._prob_stats['count']} avg_benign={avg_benign:.6f} "
            s += "avg_poison=[" + ", ".join([f"{v:.6f}" for v in avg_poison]) + "]"
            prints(s)

        return loss


    def train(self, epoch: int, optimizer: Optimizer = None, **kwargs):
        """
        Thin wrapper that delegates training to the underlying model's _train method
        using the probabilistic loss implemented in `prob_loss`.
        """
        return self.model._train(epoch=epoch, optimizer=optimizer, loss_fn=self.prob_loss,
                                 validate_fn=self.validate_fn, get_data_fn=self.get_data,
                                 save_fn=self.save, **kwargs)

    def correctness(self, keep_org=False, poison_label=True, which=0, **kwargs):
        loader = kwargs.get('loader')
        if loader is None:
            loader = self.dataset.loader['valid']
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
        _idx = set(to_list(feats_list.argsort(descending=True))[:length])
        poison_idx = set(to_list(poison_feats_list.argsort(descending=True))[:length])
        jaccard_idx = len(_idx & poison_idx) / len(_idx | poison_idx)
        return jaccard_idx
