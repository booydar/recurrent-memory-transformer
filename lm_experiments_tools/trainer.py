import contextlib
import itertools
import json
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import get_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from lm_experiments_tools.utils import rank_0, get_fn_param_names

import accelerate
from accelerate.logging import get_logger
logger = get_logger('')


@dataclass
class TrainerArgs:
    model_path: Optional[str] = field(
        default=None,
        metadata={'help': 'path where to save model (default: None)'})
    log_interval: Optional[int] = field(
        default=None,
        metadata={'help': 'log to report loss, metrics on training data every N batches (default: None)'})
    valid_interval: Optional[int] = field(
        default=None,
        metadata={'help': 'log on validation data every N batches (default: None)'})
    save_interval: Optional[int] = field(
        default=None,
        metadata={'help': 'save model every N steps (default: None)'})
    save_best: bool = field(
        default=False,
        metadata={'help': 'Save best checkpoint if validation set is provided (default: False)'})
    use_generate_on_valid: bool = field(
        default=False,
        metadata={'help': 'Use model.generate method when running validation step (default: False)'})
    # load model args
    init_checkpoint: Optional[str] = field(
        default=None,
        metadata={'help': 'path to init checkpoint to load a model from (default: None).'})
    skip_used_data: bool = field(
        default=False,
        metadata={'help': 'skip batches that were already seen by init_checkpoint (default: False)'})
    reset_lr: bool = field(
        default=False,
        metadata={'help': 'Do not load lr_scheduler from checkpoint and setup new lr (default: False)'})
    reset_iteration: bool = field(
        default=False,
        metadata={'help': 'Do not load iteration number from checkpoint and set it to 0 (default: False)'})
    reset_optimizer: bool = field(
        default=False,
        metadata={'help': 'Do not load optimizer from checkpoint and setup a new one. It might help for continuing '
                          'training from ckpt saved from fp16 O2. Otherwise loss spikes might happen (default: False)'})
    # training args
    lr: Optional[float] = field(
        default=None,
        metadata={'help': 'learning rate (default: None)'})
    batch_size: int = field(
        default=1,
        metadata={'help': 'input batch size for training (default: 1)'})
    iters: int = field(
        default=1,
        metadata={'help': 'number of training steps (i.e., gradient updates) (default: 100)'})
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={'help': 'number of batches to accumulate gradients for each worker, it multiplies total batch size.'})
    clip_grad_norm: Optional[float] = field(
        default=None,
        metadata={'help': 'torch.nn.utils.clip_grad_norm_ max_norm parameter. 0 or None is no clip (default: None)'})
    clip_grad_value: Optional[float] = field(
        default=None,
        metadata={'help': 'torch.nn.utils.clip_grad_value_ clip_value parameter. 0 or None is no clip (default: None)'})
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={'help': 'stop training if `early_stopping_patience` subsequent evalutations did not improve value of '
                          '`optimize_metric` on validation set (default: None)'})
    # scheduler args
    lr_scheduler: Optional[str] = field(
        default=None,
        metadata={'help': 'scheduler name from transformers.optimization: linear, cosine, cosine_with_restarts, '
                          'polynomial, constant, constant_with_warmup (default: None)'})
    num_warmup_steps: Optional[int] = field(
        default=None,
        metadata={'help': 'number of warming steps to get to lr (default: None)'})
    num_training_steps: Optional[int] = field(
        default=None,
        metadata={'help': 'number of training steps for scheduler, if not set iters will be used (default: None)'})
    # LRReduceOnPlateau args
    use_lr_drop: bool = field(
        default=False,
        metadata={'help': 'Enable ReduceLROnPlateau scheduler in addition to --lr_scheduler (default: False)'})
    lr_drop_factor: float = field(
        default=0.1,
        metadata={'help': 'torch.optim.lr_scheduler.ReduceLROnPlateau drop parameter. (default: 0.1)'})
    lr_drop_patience: int = field(
        default=10,
        metadata={'help': 'torch.optim.lr_scheduler.ReduceLROnPlateau patience parameter. (default: 10)'})
    lr_drop_threshold: float = field(
        default=1e-04,
        metadata={'help': 'torch.optim.lr_scheduler.ReduceLROnPlateau threshold parameter. (default: 1e-04)'})
    lr_drop_threshold_mode: str = field(
        default='rel',
        metadata={'help': 'torch.optim.lr_scheduler.ReduceLROnPlateau threshold_mode parameter. (default: rel)'})
    lr_drop_cooldown: int = field(
        default=0,
        metadata={'help': 'torch.optim.lr_scheduler.ReduceLROnPlateau cooldown parameter. (default: 0)'})
    lr_drop_min_lr: float = field(
        default=0.0,
        metadata={'help': 'torch.optim.lr_scheduler.ReduceLROnPlateau min_lr parameter. (default: 0.0)'})
    lr_drop_eps: float = field(
        default=1e-08,
        metadata={'help': 'torch.optim.lr_scheduler.ReduceLROnPlateau threshold_mode parameter. (default: 1e-08)'})
    # metrics args
    optimize_metric: str = field(
        default='loss',
        metadata={'help': 'metric name to optimize on validation set, save the best model, drop lr (default: loss)'})
    optimize_mode: str = field(
        default='min',
        metadata={'help': 'metric should be minimized (min) or maximized (max) (default: min)'})


class Trainer:
    def __init__(self, args, accelerator, model, optimizer, train_dataloader, valid_dataloader, train_sampler=None,
                 batch_transform_fn=None,
                 batch_metrics_fn=lambda _, y: {'loss': y['loss']},
                 keep_for_metrics_fn=None,
                 metrics_fn=None,
                 forward_kwargs={},
                 generate_kwargs={},
                 ) -> None:
        """Implements training loop with horovod multi-gpu, apex fp16 & grad accumulation support.

        Trainer logs all metrics returned by batch_metrics_fn and metrics_fn.

        Args:
            args: TrainerArgs passed from CLI
            model: torch model to train, model should be compatible with HF interface:
                # batch = batch_transform_fn(batch)
                output = model(**batch, **forward_kwargs)
                loss = output['loss']
            optimizer: torch optimizer
            train_dataloader (torch.utils.data.DataLoader): train set torch dataloader, distributed-aware.
            valid_dataloader (Optional(torch.utils.data.DataLoader)]): validation set torch dataloader,
                distributed-aware, optional.
            batch_transform_fn (Optional): function to be applied to the output from DataLoader, should be used to
                create inputs compatible (if not already) with training model, e.g.:
                    f(batch) -> {'input_ids': ..., 'attention_mask': ..., 'labels': ..., ...}.
            batch_metrics_fn (Optional): function f(batch, model_output) to compute batch-lvl metrics.
                Metrics are averaged across batches: avg_i(metric(batch_i, labels_i)),
                not metric([batch_1; batch_2; ...], labels). batch_metrics_fn could be used for computing loss, metrics
                on large datasets, pre-training, where exact metrics values are not so important or computing exact
                metrics is resource-exhaustive.
                Should return dict: {'metric_name': metric_value, ...}
            keep_for_metrics_fn (Optional): f(batch, model_output) to keep predictions, labels or other data that would
                be used to compute metrics on full validation set and every log_interval on train set.
                Should return dict {'key_1': tensor/np.array/scalar/list, 'key_2': ...}.
                The result of keep_for_metrics_fn will be aggregated into tensor/list and passed to metrics_fn.
                Check `collect_metrics` function for further details.
            metrics_fn (Optional): f(metrics_data) to compute metrics based on values stored by keep_for_metrics_fn.
                Should return dict: {'metric_name': metric_value, ...}
            forward_kwargs (Optional): keyworded arguments that should be passed to model.__call___ along with **batch.
                `batch` should be used to pass Tensors and **kwargs should be used to pass some flags or other
                arguments independent from batch size.
            generate_kwargs (Optional): keyworded arguments that should be passed to model.geberate along with
                `input_ids`.
        """
        # we assume that train/valid/test dataloaders are already multi-gpu aware
        self.accelerator = accelerator
        logger.info(f'setting up trainer with accelerator state:\n{self.accelerator.state}')
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.train_sampler = train_sampler
        self.valid_dataloader = valid_dataloader
        self.batch_transform_fn = batch_transform_fn
        self.batch_metrics_fn = batch_metrics_fn
        self.keep_for_metrics_fn = keep_for_metrics_fn
        self.metrics_fn = metrics_fn
        self.forward_kwargs = deepcopy(forward_kwargs)
        self.generate_kwargs = deepcopy(generate_kwargs)

        self.device = self.accelerator.device

        self.args = args

        self.per_worker_batch_size = self.args.batch_size * self.args.gradient_accumulation_steps
        self.global_batch_size = self.per_worker_batch_size * self.accelerator.num_processes

        self.model_forward_args = set(get_fn_param_names(self.accelerator.unwrap_model(self.model).forward))

        if self.args.clip_grad_norm is not None and self.args.clip_grad_value is not None:
            raise RuntimeError(f'Only one from clip_grad_norm and clip_grad_value should be set, but found '
                               f'clip_grad_norm = {self.args.clip_grad_norm}, '
                               f'clip_grad_value = {self.args.clip_grad_value}.')

        self.clip_grad = False
        if self.args.clip_grad_norm or self.args.clip_grad_value:
            self.clip_grad = True

        self.args.optimize_mode = getattr(self.args, 'optimize_mode', 'min')
        self.args.optimize_metric = getattr(self.args, 'optimize_metric', 'loss')
        if self.args.optimize_mode == 'min':
            self.metric_improved_fn = lambda old_m, new_m: old_m > new_m
        else:
            self.metric_improved_fn = lambda old_m, new_m: old_m < new_m
        self.early_stopping_counter = 0

        self.tb = None
        # write tensorboard logs only from rank 0 and if model_path is specified
        if self.accelerator.is_main_process and self.args.model_path is not None:
            self.tb = SummaryWriter(log_dir=self.args.model_path)

        # move model to gpu
        self.model.to(self.device)

        if args.lr_scheduler:
            if args.lr is None:
                raise RuntimeError('Set learning_rate to use learning rate schedulers.')
            if args.num_training_steps is None:
                args.num_training_steps = args.iters
            self.lr_scheduler = get_scheduler(args.lr_scheduler, self.optimizer,
                                              args.num_warmup_steps, args.num_training_steps)
            # todo: do we need to prepare scheduler with accelerate?
            self.accelerator.register_for_checkpointing(self.lr_scheduler)
        else:
            self.lr_scheduler = None

        self.args.use_lr_drop = getattr(self.args, 'use_lr_drop', False)
        if self.args.use_lr_drop and self.lr_scheduler is not None:
            raise RuntimeError('lr drop can not be used with other lr schedulers')
        if self.args.use_lr_drop and self.valid_dataloader is None:
            raise RuntimeError('lr drop is based on validation metrics, but validation set is not set')
        if self.args.use_lr_drop:
            self.lr_drop_scheduler = ReduceLROnPlateau(self.optimizer, mode=self.args.optimize_mode,
                                                       factor=self.args.lr_drop_factor,
                                                       patience=self.args.lr_drop_patience,
                                                       threshold=self.args.lr_drop_threshold,
                                                       threshold_mode=self.args.lr_drop_threshold_mode,
                                                       cooldown=self.args.lr_drop_cooldown,
                                                       min_lr=self.args.lr_drop_min_lr,
                                                       eps=self.args.lr_drop_eps,
                                                       verbose=True)
        else:
            self.lr_drop_scheduler = None

        self.n_iter = 0
        self.n_epoch = 0
        # self.batch_metrics keeps per-batch metrics for all batches in log_interval
        self._reset_batch_metrics()
        # self.metrics_data stores all intermediate batches data (in log_interval) to be used lately to compute metrics
        self._reset_metrics_data()
        # self.metrics keeps the last logged metrics
        self._reset_metrics()
        if self.args.init_checkpoint:
            self.load(args.init_checkpoint, self.args.reset_optimizer, self.args.reset_lr, self.args.reset_iteration)

    def step(self, batch, is_train_mode=True) -> Tuple[Dict[str, float], Dict[str, list]]:
        """Performs one step (forward and optionally backward and optimizer.step()) over data in a batch.

        Batch is splitted on sub-batches of self.args.batch_size size, loss and gradients are accumulated.

        Args:
            batch (dict): dict with inputs, inputs_mask, targets, & all the data that is required by model.forward()
            is_train_mode (bool, optional): In train mode we compute gradients, do backprop and optimizer.step().
                Defaults to True.

        Returns:
            float: loss on batch
        """
        if is_train_mode:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        if self.batch_transform_fn:
            batch = self.batch_transform_fn(batch)

        batch_sizes = []
        for k in batch:
            # filter keys in batch to pass to model only supported arguments
            if k in self.model_forward_args:
                batch[k] = batch[k].to(self.device)
                batch_sizes += [batch[k].size(dim=0)]
        if not np.all(np.array(batch_sizes) == batch_sizes[0]):
            raise RuntimeError(f'not all elements in a batch have equal dim 0 size: {batch_sizes}')
        batch_size = batch_sizes[0]

        batch_metrics = defaultdict(lambda: 0.0)
        batch_metrics_data = defaultdict(lambda: [])
        with torch.set_grad_enabled(is_train_mode):
            for j in range(0, batch_size, self.args.batch_size):
                is_last_batch = (j == (batch_size // self.args.batch_size - 1) * self.args.batch_size)
                grad_sync_context = contextlib.nullcontext if is_last_batch else self.accelerator.no_sync
                with grad_sync_context(self.model):
                    subbatch = {k: batch[k][j: j + self.args.batch_size] for k in batch}
                    # filter items from batch that are not used by model forward
                    outputs = self.model(**{k: subbatch[k] for k in subbatch if k in self.model_forward_args},
                                         **self.forward_kwargs)
                    loss = outputs['loss']
                    # divide loss on gradient_accumulation_steps to get average loss for sub-batches
                    # no need, accelerate does it internally (need to pass gradient_accumulation_steps to accelerator)
                    # loss = loss / self.args.gradient_accumulation_steps

                    if not is_train_mode and self.args.use_generate_on_valid:
                        generate_kwargs = deepcopy(self.generate_kwargs)
                        if 'max_length' not in generate_kwargs and 'labels' in subbatch:
                            # if max_length is not set and labels are in subbatch, generate to the length of labels+1
                            # +1 as special tokens could be generated by the model
                            generate_kwargs['max_length'] = subbatch['labels'].shape[-1] + 1
                        if 'attention_mask_generate' in subbatch:
                            generate_kwargs['attention_mask'] = subbatch['attention_mask_generate'].to(self.device)
                        elif 'attention_mask' in subbatch:
                            generate_kwargs['attention_mask'] = subbatch['attention_mask']
                        if 'global_attention_mask' in subbatch:
                            generate_kwargs['global_attention_mask'] = subbatch['global_attention_mask']
                        if 'input_ids_generate' in subbatch:
                            generation_outputs = self.accelerator.unwrap_model(self.model).generate(subbatch['input_ids_generate'].to(self.device), **generate_kwargs)
                        else:
                            generation_outputs = self.accelerator.unwrap_model(self.model).generate(subbatch['input_ids'], **generate_kwargs)
                        outputs['generation_outputs'] = generation_outputs

                    metrics = self.batch_metrics_fn(subbatch, outputs)

                    for k in metrics:
                        metrics[k] = metrics[k] / self.args.gradient_accumulation_steps
                        if isinstance(metrics[k], torch.Tensor):
                            metrics[k] = metrics[k].detach().item()
                        batch_metrics[k] += metrics[k]

                    if self.keep_for_metrics_fn and self.metrics_fn:
                        for k, v in self.keep_for_metrics_fn(subbatch, outputs).items():
                            batch_metrics_data[k] += [v.detach().cpu() if isinstance(v, torch.Tensor) else v]

                    if is_train_mode:
                        # backward
                        self.accelerator.backward(loss)

            # all gradients are collected and synced
            if is_train_mode:
                # log gradients norm, clip gradients and perform opt.step(), lr_scheduler.step()
                if self.clip_grad:
                    global_grad_norm = self._clip_gradients()
                else:
                    global_grad_norm = self._get_gradients_global_norm()
                # track clipped grad norms
                self.global_grad_norms += [global_grad_norm]

                self.optimizer.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()
        return batch_metrics, batch_metrics_data

    def _clip_gradients(self):
        # accelerate recommends to use accelerator.clip_grad_norm_
        # it unscales gradients internally and makes some checks for different distributed setups.
        # However, in this case we loose access to unscaled gradients before clipping (to log them).
        # Also, deepspeed implements clipping internally and does not return grad norm
        # (todo: check deepspeed.utils.safe_get_full_grad(param))
        params = self.model.parameters()
        grad_norm = 0.0
        if self.args.clip_grad_value:
            self.accelerator.clip_grad_value_(params, self.args.clip_grad_value)
            grad_norm = self._get_gradients_global_norm()
        elif self.args.clip_grad_norm:
            grad_norm = self.accelerator.clip_grad_norm_(params, self.args.clip_grad_norm)
            grad_norm = grad_norm.item() if grad_norm is not None else 0.0
        return grad_norm

    def _get_gradients_global_norm(self):
        # get gradients global norm (in the same way as in torch.nn.utils.clip_grad_norm_)
        params = self.model.parameters()
        params = [p for p in params if p.grad is not None]
        if len(params) == 0:
            return 0.0
        total_norm = torch.linalg.norm(torch.stack([torch.linalg.norm(p.grad.detach()) for p in params])).item()
        return total_norm

    def _train_batch_generator(self):
        while self.n_iter <= self.args.iters:
            if self.train_sampler:
                self.train_sampler.set_epoch(self.n_epoch)
            # self.train_dataloader
            for batch in self.train_dataloader:
                if self.n_iter > self.args.iters:
                    return
                yield batch
                self.n_iter += 1
            self.n_epoch += 1

    def _skip_n_train_batches(self, train_batches, n):
        # we have to re-iterate over dataset
        # currently, skipping is based on number of iterations, not samples seen on previous run:
        #   (n_gpus x bs x n_grad_acc x n_iters)
        # todo: save number of seen samples in checkpoint
        logger.info(f'Skipping {n} batches from the dataset from epoch {self.n_epoch}...')
        # skipping...
        for _ in tqdm(itertools.islice(train_batches, n), disable=(not self.accelerator.is_main_process),
                      desc='Skipping...', total=n):
            ...

    def _add_batch_metrics(self, batch_metrics: Dict[str, Union[float, torch.Tensor]], split: str):
        """Adds metrics values for batch-lvl metrics.

        Args:
            split (str): train / valid
            batch_metrics (Dict[str, Union[float, torch.Tensor]]): batch-lvl metrics values, scalars.
        """
        for k in batch_metrics:
            self.batch_metrics[split][k] += [batch_metrics[k]]

    def _add_metrics_data(self, metrics_data: Dict[str, torch.Tensor], split: str):
        """Adds metrics data to keep. These data would be used to compute metrics later with get_metrics.

        Args:
            split (str): train / valid
            value (Dict[str, torch.Tensor]): dict with metrics data, data[name].shape[0] is batch size.
        """
        for k in metrics_data:
            self.metrics_data[split][k] += metrics_data[k]

    def _reset_batch_metrics(self, split=None):
        if split is None:
            # e.g., self.batch_metrics['train']['metric_name'] is a list of metric values
            self.batch_metrics = defaultdict(lambda: defaultdict(list))
        else:
            self.batch_metrics[split] = defaultdict(list)

    def _reset_metrics_data(self, split=None):
        if split is None:
            self.metrics_data = defaultdict(lambda: defaultdict(list))
        else:
            self.metrics_data[split] = defaultdict(list)

    def _reset_metrics(self, split=None):
        if split is None:
            self.metrics = dict()
        else:
            del self.metrics[split]

    def collect_metrics(self, split: str) -> dict:
        """
        Collects batch-lvl metrics from batch_metrics_fn and computes metrics with metrics_fn on data collected from
        keep_for_metrics_fn. Once the metrics are collected we drop everything that was previously collected.

        Args:
            split (str): data split name train/valid for which metrics should be collected

        Returns:
            dict: dictionary with collected metrics
        """
        # batch-lvl metrics
        metrics = {}
        # collect metrics names from all processes: it is possible that different workers might have different
        # set of metrics (e.g., some metric could be not available for some batches).
        metrics_keys = set(accelerate.utils.gather_object(list(self.batch_metrics[split].keys())))
        if metrics_keys != self.batch_metrics[split].keys():
            missing_metrics_keys = metrics_keys - self.batch_metrics[split].keys()
            logger.warning(f'some of the batch-lvl metrics on rank_{self.accelerator.process_index} are missing, '
                           f'but were found on another ranks: {missing_metrics_keys}')
        metrics_keys = sorted(metrics_keys)
        for k in metrics_keys:
            metrics[k] = accelerate.utils.gather_object(self.batch_metrics[split][k])
            metrics[k] = np.mean(metrics[k])
        # compute metrics from metrics data
        if self.keep_for_metrics_fn and self.metrics_fn:
            metrics_data = {}
            data_keys = set(accelerate.utils.gather_object(list(self.metrics_data[split].keys())))
            if data_keys != self.metrics_data[split].keys():
                missing_data_keys = data_keys - self.metrics_data[split].keys()
                logger.warning(f'some of the data collected from keep_for_metrics_fn on '
                               f'rank_{self.accelerator.process_index} is missing, '
                               f'but was found on another ranks: {missing_data_keys}')
            data_keys = sorted(data_keys)
            for k in data_keys:
                metrics_data[k] = accelerate.utils.gather_object(self.metrics_data[split][k])
                m_shape = getattr(metrics_data[k][0], 'shape', None)
                if m_shape is None:
                    # data is not a tensor, collect it into python list
                    metrics_data[k] = list(chain.from_iterable(metrics_data[k]))
                elif len(m_shape) == 0:
                    # if scalars
                    metrics_data[k] = torch.stack(metrics_data[k])
                elif all(m_shape[1:] == t.shape[1:] for t in metrics_data[k]):
                    # concat tensors if all shapes are equal except the first
                    metrics_data[k] = torch.cat(metrics_data[k])
                else:
                    # can't concat tensors with diff last shapes, so collecting them into python list
                    metrics_data[k] = list(chain.from_iterable([t.tolist() for t in metrics_data[k]]))
            m = self.metrics_fn(metrics_data)
            if len(metrics.keys() & m.keys()) != 0:
                logger.warning(f'metrics ({m.keys()}) and batch-lvl metrics ({metrics.keys()}) have common names. '
                               f'Batch-lvl metric value would be overwritten.')
            metrics.update(m)
        self._reset_batch_metrics(split)
        self._reset_metrics_data(split)
        self.metrics[split] = metrics
        return metrics

    def train(self) -> None:
        pbar = tqdm(total=self.args.iters, desc='Train', disable=(not self.accelerator.is_main_process))
        pbar.update(self.n_iter)

        train_batches = self._train_batch_generator()

        # skip used data if needed
        if self.args.skip_used_data and self.n_iter > 0:
            train_size = None
            try:
                train_size = len(self.train_dataloader)
            except TypeError as e:
                logger.info(f"Can't get train_dataloader length:\n{e}")
            # if we know train_size and number of epochs passed -> jump to this epoch and re-iterate over remainders
            skip_iter = self.n_iter % train_size if train_size else self.n_iter
            self.n_iter = (self.n_iter // train_size) * train_size if train_size else 0
            self._skip_n_train_batches(train_batches, skip_iter)

        self._reset_batch_metrics('train')
        self._reset_metrics_data('train')
        self.global_grad_norms = []
        best_valid_metric = np.inf if self.args.optimize_mode == 'min' else -np.inf
        valid_metric = best_valid_metric
        valid_loss = np.inf
        train_loss = np.inf
        self.early_stopping_counter = 0
        for batch in train_batches:
            iteration_start = time.time()
            batch_metrics, batch_metrics_data = self.step(batch, is_train_mode=True)
            iteration_time = time.time() - iteration_start
            self._add_batch_metrics(batch_metrics, split='train')
            if self.keep_for_metrics_fn and self.metrics_fn:
                self._add_metrics_data(batch_metrics_data, split='train')

            # logging
            if self.args.log_interval and self.n_iter % self.args.log_interval == 0:
                # batch-lvl averaged metrics:
                train_metrics = self.collect_metrics(split='train')
                train_loss = train_metrics['loss']
                global_grad_norms = accelerate.utils.gather_object(self.global_grad_norms)
                self.global_grad_norms = []
                if self.accelerator.is_main_process:
                    # todo: move logging, move to self.log()
                    for k in train_metrics:
                        logger.info(f'step: {self.n_iter}/{self.args.iters} {k}: {train_metrics[k]:.4f}')
                        if self.tb:
                            self.tb.add_scalar(f'{k}/iterations/train', train_metrics[k], self.n_iter)
                            self.tb.add_scalar(f'{k}/samples/train', train_metrics[k],
                                               self.n_iter * self.global_batch_size)
                    # log iteration time
                    if self.tb:
                        self.tb.add_scalar('time/iterations/per_iter', iteration_time, self.n_iter)
                        self.tb.add_scalar('time/samples/per_iter', iteration_time,
                                           self.n_iter * self.global_batch_size)
                    # log learning rate
                    for j, param_group in enumerate(self.optimizer.param_groups):
                        # adafactor uses external lr to compute its own lr if scale_parameter is true
                        # adafactor might not have external lr in case if relative_step is used
                        for p in ['lr', 'scaled_lr']:
                            if p in param_group and param_group[p] is not None and self.tb:
                                self.tb.add_scalar(f'{p}/iterations/param_group_{j}', param_group[p], self.n_iter)
                                self.tb.add_scalar(f'{p}/samples/param_group_{j}', param_group[p],
                                                   self.n_iter * self.global_batch_size)
                    # log gradients global norm
                    gnorm = np.mean(global_grad_norms) if len(global_grad_norms) > 0 else 0
                    if self.tb:
                        self.tb.add_scalar('gradients_global_norm/iterations', gnorm, self.n_iter)
                        self.tb.add_scalar('gradients_global_norm/samples', gnorm, self.n_iter * self.global_batch_size)

            # validation
            if self.valid_dataloader is not None and self.n_iter % self.args.valid_interval == 0:
                # todo: we can use other metrics than loss here
                valid_metrics = self.validate(self.valid_dataloader)
                valid_loss = valid_metrics['loss']
                valid_metric = valid_metrics[self.args.optimize_metric]
                if self.metric_improved_fn(best_valid_metric, valid_metric):
                    best_valid_metric = valid_metric
                    self.early_stopping_counter = 0
                    logger.info(f'The best {self.args.optimize_metric} metric was improved to: {best_valid_metric}')
                    if self.args.save_best:
                        self.save(self.args.model_path, suffix='best')
                else:
                    self.early_stopping_counter += 1
                    logger.info(f'Metric was not improved for the last #{self.early_stopping_counter} evaluations')
                if self.accelerator.is_main_process and self.tb:
                    self.tb.add_scalar('patience/iterations', self.early_stopping_counter, self.n_iter)
                    self.tb.add_scalar('patience/samples', self.early_stopping_counter,
                                       self.n_iter * self.global_batch_size)
                if self.lr_drop_scheduler:
                    self.lr_drop_scheduler.step(valid_metric)

            # saving model
            if self.args.save_interval and self.n_iter % self.args.save_interval == 0:
                self.save(self.args.model_path)

            pbar.update(1)
            pbar.set_postfix({'train_loss': f'{train_loss:.3f}',
                              'valid_loss': f'{valid_loss:.3f}',
                              f'best_valid_{self.args.optimize_metric}': f'{best_valid_metric:.3f}'
                              })

            if self.args.early_stopping_patience is not None and \
                    self.early_stopping_counter > self.args.early_stopping_patience:
                logger.info('Early stopping triggered: stopping training...')
                break

        # clean-up
        pbar.close()
        if self.accelerator.is_main_process and self.tb:
            self.tb.flush()
        logger.info('Done!')

    def validate(self, dataloader, split='valid', write_tb=True) -> Dict[str, float]:
        logger.info(f'start validation at step {self.n_iter}')
        self._reset_batch_metrics(split)
        self._reset_metrics_data(split)

        n_valid_batches = None
        try:
            n_valid_batches = len(dataloader)
        except TypeError:
            # in case if dataset has no len() method (IterableDataset?)
            n_valid_batches = None

        pbar = tqdm(total=n_valid_batches, desc='Validation', disable=(not self.accelerator.is_main_process))
        for batch in dataloader:
            batch_metrics, batch_metrics_data = self.step(batch, is_train_mode=False)
            self._add_batch_metrics(batch_metrics, split=split)
            if self.keep_for_metrics_fn and self.metrics_fn:
                self._add_metrics_data(batch_metrics_data, split=split)
            pbar.update()
        pbar.close()

        metrics = self.collect_metrics(split=split)
        if self.accelerator.is_main_process:
            # todo: separate logging from validation/training
            for k in metrics:
                # print(f'Validation on {split} {k}: {metrics[k]:.4f}')
                logger.info(f'Validation on {split} {k}: {metrics[k]:.4f}')
                if self.tb and write_tb:
                    self.tb.add_scalar(f'{k}/iterations/{split}', metrics[k], self.n_iter)
                    self.tb.add_scalar(f'{k}/samples/{split}', metrics[k], self.n_iter * self.global_batch_size)
            if self.tb and write_tb:
                self.tb.flush()

        return metrics

    def load(self, load_path: Union[str, Path], reset_optimizer=False, reset_lr=False, reset_iteration=False) -> None:
        """Loads model, trainer state, and accelerate state.

        Args:
            load_path (Union[str, Path]): Path to the folder with model ckpt, trainer state and accelerate state.
                OR path to the ckpt file.
            reset_optimizer (bool, optional): _description_. Defaults to False.
            reset_lr (bool, optional): _description_. Defaults to False.
            reset_iteration (bool, optional): _description_. Defaults to False.
        """
        load_path = Path(load_path)
        load_only_model_ckpt = (reset_optimizer and reset_lr) or load_path.is_file()

        # load trainer state if there is one
        trainer_state = {}
        trainer_state_path = None
        if load_path.is_dir():
            trainer_state_path = load_path / 'trainer.pckl'
        # check if trainer state is in the same folder as ckpt
        elif load_path.is_file() and (load_path.parent / 'trainer.pckl').exists():
            trainer_state_path = load_path.parent / 'trainer.pckl'
        if trainer_state_path:
            trainer_state = torch.load(trainer_state_path, map_location='cpu')
        if not reset_iteration:
            self.n_iter = trainer_state.get('iteration', 0) + 1  # as saved iteration is already performed
            self.n_epoch = trainer_state.get('epoch', 0)

        if not load_only_model_ckpt:
            logger.info('Loading model, trainer, and accelerate state')
            self.accelerator.load_state(load_path / 'accelerate_state')
            if reset_optimizer:
                raise RuntimeError('Reset optimizer only is not supported. You may load only model weights with'
                                   '--reset_optimizer --reset_lr')
            if reset_lr:
                raise RuntimeError('Reset lr & scheduler only is not supported. You may load only model weights with'
                                   '--reset_optimizer --reset_lr')
        else:
            logger.info(f'Loading model from {load_path}')
            checkpoint = torch.load(load_path, map_location='cpu')
            missing_k, unexpected_k = self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint, strict=False)
            if len(missing_k) != 0:
                logger.info(f'{missing_k} were not loaded from checkpoint! These parameters were randomly initialized.')
            if len(unexpected_k) != 0:
                logger.info(f'{unexpected_k} were found in checkpoint, but model is not expecting them!')
            del checkpoint
        logger.info(f'Start iteration = {self.n_iter}')

    def save(self, save_path, suffix='') -> None:
        if save_path is not None:
            if suffix == '':
                save_path = f'{save_path}/model_{self.n_iter}'
            else:
                save_path = f'{save_path}/model_{suffix}'

            self.accelerator.save_state(f'{save_path}/accelerate_state')
            self.accelerator.save_model(self.model, f'{save_path}')
            self.save_metrics(save_path)

            if self.accelerator.is_main_process:
                to_save = {
                    # handled by accelerate
                    # 'model_state_dict': self.accelerator.get_state_dict(self.model),
                    # 'optimizer_state_dict': self.optimizer.state_dict(),
                    'iteration': self.n_iter,
                    'epoch': self.n_epoch,
                    'metrics': self.metrics}
                # handled by accelerate
                # if self.use_torch_amp:
                #     to_save['torch_amp'] = self.amp_grad_scaler.state_dict()
                # if self.lr_scheduler:
                #     to_save['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
                torch.save(to_save, f'{save_path}/trainer.pckl')
            logger.info(f'Model, trainer, and accelerate state were saved to {save_path}')

    @rank_0
    def save_metrics(self, save_path) -> None:
        """Saves all metrics into metrics.json
        After trainer.train(...) you might want to load the best checkpoint and validate it on some test sets, e.g.:
            trainer.validate(valid, split='valid')
            trainer.validate(test_1, split='test_1')
            trainer.validate(test_2, split='test_2')
        and save metrics into a file, e.g.:
            trainer.save_metrics(save_path=args.model_path)
        """
        if save_path is not None:
            save_path = f'{save_path}/metrics.json'
            for split in self.metrics:
                for k in self.metrics[split]:
                    if isinstance(self.metrics[split][k], torch.Tensor):
                        self.metrics[split][k] = self.metrics[split][k].numpy().tolist()
                    if isinstance(self.metrics[split][k], np.ndarray):
                        self.metrics[split][k] = self.metrics[split][k].tolist()
            try:
                json.dump(self.metrics, open(save_path, 'w'), indent=4)
            except TypeError as e:
                logger.warning(f'Unable to save metrics: {e}.\nmetrics: {self.metrics}')
