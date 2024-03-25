import json
import logging
import os
import math
import shutil
from pathlib import Path
from itertools import chain

# from dotenv import load_dotenv
import torch
import numpy as np
import datasets
import transformers
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

from lm_experiments_tools.trainer import Trainer, TrainerArgs

from torch.nn.utils.rnn import pad_sequence

import accelerate

# load_dotenv()

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')


# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

# import transformers  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser  # noqa: E402

from lm_experiments_tools.utils import get_cls_by_name, get_optimizer, prepare_run  # noqa: E402
import lm_experiments_tools.optimizers as optimizers  # noqa: E402

# limit # of CPU threads to be used per pytorch worker, otherwise it might use all cpus and throttle gpus
# > 2 fails cause of https://github.com/pytorch/pytorch/issues/56615
# need to upgrade to torch>1.8.1
# torch.set_num_threads(4)
# all gpus set with CUDA_VISIBLE_DEVICES are visible to process, indexing from 0 to ...

parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--task_name', type=str, help='Scrolls task name: "gov_report", "summ_screen_fd", "qmsum", '
                                                  '"narrative_qa", "qasper", "quality", "contract_nli"')
parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--show_valid_examples', type=int, default=0,
                    help='how many valid examples to show during training (default: 0)')
# parser.add_argument('--input_seq_len', type=int, default=128, help='input sequnce length (default: 128).')
# parser.add_argument('--target_seq_len', type=int, default=16, help='target sequnce length, should be set to '
                                                                #    'max(len(target))+1 for EOS (default: 16).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

parser.add_argument('--input_prefix', type=str, default='', help='add task prefix to an input string (default: "")')
parser.add_argument('--sliding_window', action='store_true', help='use slinding window attention mask, '
                    'eval on last segment only', default=False)

# model args
parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "")')
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: "")')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')
parser.add_argument('--memory_cell_cls', type=str, default=None, help='cell class for RMT')
parser.add_argument('--recurrent_wrapper_cls', type=str, default=None, help='recurrent wrapper class for RMT')
parser.add_argument('--model_cpt', type=str, default=None, help='pretrained model checkpoint path')
parser.add_argument('--model_type', type=str, default='encoder-decoder',
                    help='model type, encoder, encoder-decoder, decoder, affects preprocessing '
                         '(default: encoder-decoder)')

# Dataset args
parser.add_argument('--key_size', type=int, default=None, help='number of digits in keys')
parser.add_argument('--value_size', type=int, default=None, help='number of digits in values')
parser.add_argument('--num_pairs', type=int, default=None, help='number of key-value pairs in sample')
parser.add_argument('--dataset_path', type=str, default="/home/jovyan/rmt/datasets/associative_retrieval/", help="path to saved datasets")
parser.add_argument('--train_size', type=int, default=10000, help='number of samples in train split')
parser.add_argument('--valid_size', type=int, default=1000, help='number of samples in validation split')
parser.add_argument('--test_size', type=int, default=2000, help='number of samples in test split')
parser.add_argument('--segment_size', type=int, default=128, help='number of useful tokens in a segment')
parser.add_argument('--rewrite_setting', action='store_true', default=False,
                    help='keys can occur several times')

# Aydar # RMT args 
parser.add_argument('--input_size', type=int, default=None, help='maximal input size of the backbone model')
parser.add_argument('--num_mem_tokens', type=int, default=None, help='number of memory tokens.')
parser.add_argument('--max_n_segments', type=int, default=1, help='maximal segment number')
parser.add_argument('--vary_n_segments', action='store_true', default=False, help='Randomly choose segment number from 1 to max_n_segments')
parser.add_argument('--segment_alignment', type=str, default=None, help="How to align segments when splitting input")
# parser.add_argument('--sum_loss', action='store_true', default=False,
#                     help='with this flag task loss from all segments is summed')
# parser.add_argument('--bptt_depth', type=int, default=-1, help='max number of previous segments in gradient computation.')
# parser.add_argument('--segment_ordering', type=str, help='segment order', default='regular',
#                     choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])
# parser.add_argument('--memory_forward_func', type=str, help='path to memory forward funсtion script', default=None)
# parser.add_argument('--memory_layers', type=str, help='memory-augmented layer inds or "all" for all layers', default=None)
# parser.add_argument('--share_memory_layers', action='store_true', help='share weights of memory layers', default=False)
# parser.add_argument('--reconstruction_loss_coef', type=float, default=None,
#                     help='reconstuction loss ratio in total loss')
# # parser.add_argument('--segment_ordering', type=str,help='????', default='regular',
# #                     choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])
# parser.add_argument('--retain_graph', action='store_true', help='Retain computation graph during backward pass', default=False)
# parser.add_argument('--use_truncated_backward', action='store_true', default=False,
#                     help='whether to use RMT truncated bptt method in backward')
# parser.add_argument('--k1', type=int, default=-1, help='(not implemented) If not -1, gradient update is done each k1 segments')
parser.add_argument('--k2', type=int, default=-1, help='number of last segments used by backward')
parser.add_argument('--freeze_model_weights', action='store_true', default=False,
                    help='Stop training all model weights except memory layers')
parser.add_argument('--backbone_cpt', type=str, default=None, help='backbone model checkpoint path')


# tokenizer
# todo: add wordpiece tokenizers support?
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay (default: 0.0)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')


NUM_SYMBOLS = 16
from tqdm.auto import tqdm

def generate_pairs(key_size, value_size, num_pairs, num_samples):
    keys = torch.empty((num_samples, num_pairs, key_size))

    if not rewrite_setting:
        for i in tqdm(range(num_samples)):
            key = torch.randperm(NUM_SYMBOLS ** key_size)[:num_pairs]
            for j in range(key_size):
                keys[i, :, j] = key % NUM_SYMBOLS
                key //= NUM_SYMBOLS
    else:
        keys = torch.randint(0, NUM_SYMBOLS, (num_samples, num_pairs, key_size))

    
    # keys = torch.randint(0, NUM_SYMBOLS, (num_pairs * 2, key_size))
    # keys[:, 0] = torch.randint(1, NUM_SYMBOLS, (num_pairs * 2, ))
    
    # unique = keys.unique(dim=0)
    # delta_pairs = num_pairs - unique.shape[0]
    # if delta_pairs > 0:
    #     print('got unique')
    #     return generate_pairs(key_size, value_size, num_pairs)

    # selected_ids = torch.randperm(unique.shape[0])[:num_pairs]
    # keys = unique[selected_ids]

    values = torch.randint(0, NUM_SYMBOLS, (num_samples, num_pairs, value_size))
    # values[:, 0] = torch.randint(1, NUM_SYMBOLS, (num_pairs, ))
    return keys, values


class ARDataset:
    def __init__(self, key_size, value_size, sample_len=1, num_samples=20_000):
        self.sample_len = sample_len
        self.keys, self.values = generate_pairs(key_size, value_size, sample_len, num_samples)
        # self.keys = keys.reshape(num_samples, -1)
        # self.values = values.reshape(num_samples, -1)
        if not rewrite_setting:
            self.target_key_inds = torch.randint(sample_len, (num_samples, ))
        else:
            self.target_key_inds = torch.empty((num_samples,), dtype=torch.long)
            for i in tqdm(range(num_samples)):
                unique_keys = self.keys[i].unique(dim=0)
                key = unique_keys[torch.randperm(len(unique_keys))[0]]
                try:
                    idx = torch.max(torch.where(torch.all(self.keys[i] == key, dim=-1))[0], dim=0)[0].long()
                except Exception:
                    print(f"{self.keys[i]}, {key}")
                    raise 1
                assert torch.all(self.keys[i][idx] == key)
                self.target_key_inds[i] = idx
    
    def __getitem__(self, idx):
        keys, values, tgt_ind = self.keys[idx], self.values[idx], self.target_key_inds[idx]
        # dim = 0 if keys.ndim == 1 else 1
        # keys = torch.chunk(keys, self.sample_len, dim=dim)
        # values = torch.chunk(values, self.sample_len, dim=dim)
        sample = {'keys': keys, 'values': values, 'target_key_ind': tgt_ind}
        return sample
    
    def __len__(self):
        return self.keys.shape[0]
    

if __name__ == '__main__':
    args = parser.parse_args()
    # set current working dir
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)

    accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    from accelerate.logging import get_logger
    logger = get_logger('')

    logger.info(f'num processes: {accelerator.num_processes}')
    logger.info(f'mixed precision: {accelerator.mixed_precision}')

    if args.model_path is None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')

    rewrite_setting = args.rewrite_setting
    # # create model path and save configuration
    # # todo: use prepare run
    # if accelerator.is_main_process and args.model_path is not None:
    #     model_path = Path(args.model_path)
    #     if not model_path.exists():
    #         Path(model_path).mkdir(parents=True)
    #     args_dict = collect_run_configuration(args)
    #     # todo: if model path exists and there is config file, write new config file aside
    #     json.dump(args_dict, open(model_path/'config.json', 'w'), indent=4)
    #     open(model_path / 'git.diff', 'w').write(get_git_diff())

    prepare_run(args, logger, logger_fmt)

    # if not args.from_pretrained:
    #     tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.model_type == 'decoder':
        block_size = args.segment_size
        sep_token, gen_token, eos_token = 100, 101, 102
        mask_token = 103

        def collate_fn(batch):
            keys = [b['keys'] for b in batch]
            values = [b['values'] for b in batch]
            tgt_inds = [b['target_key_ind'].item() for b in batch]

            bs = len(keys)
            sep_tokens = torch.ones(bs, 1) * sep_token
            eos_tokens = torch.ones(bs, 1) * eos_token
            gen_tokens = torch.ones(bs, 1) * gen_token
            masked_tokens = torch.ones(bs, args.value_size) * mask_token
            sample = []

            for i in range(args.num_pairs):
                sample.append(torch.stack([k[i] for k in keys]))
                sample.append(sep_tokens)
                sample.append(torch.stack([v[i] for v in values]))
                sample.append(eos_tokens)

            target_keys = torch.stack([k[i] for i, k in zip(tgt_inds, keys)])
            target_values = torch.stack([k[i] for i, k in zip(tgt_inds, values)])

            sample.append(target_keys)
            sample.append(gen_tokens)
            sample.append(masked_tokens)
            sample.append(eos_tokens)

            sample.append(target_keys)
            sample.append(gen_tokens)

            input_ids_generate = torch.cat(sample, dim=1)

            sample.append(target_values)
            sample.append(eos_tokens)
            input_ids = torch.cat(sample, dim=1)

            labels_mask = torch.zeros_like(input_ids).bool()
            labels_mask[:, -args.value_size - 2:] = True

            collated = {'input_ids': input_ids.long(), 
                        'input_ids_generate': input_ids_generate.long(), 
                        'attention_mask': torch.ones_like(input_ids).bool(),
                        'attention_mask_generate': torch.ones_like(input_ids_generate).bool(),
                        'labels': input_ids.long(), 
                        'labels_mask': labels_mask, 
                        }
            
            # print(torch.chunk(collated['input_ids'][:4], args.num_pairs + 2, 1))
            return collated
            
    else:
        raise NotImplementedError(f'Unknown model type {args.model_type}')

    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
    # get train dataset
    logger.info(f'preparing dataset for: {args.task_name}')
    
    dataset_name = f"AR_k{args.key_size}_v{args.value_size}_p{args.num_pairs}"
    path = os.path.join(args.dataset_path, dataset_name)

    if os.path.exists(path):
        print(f"Loading {dataset_name} from disk.")
        train_dataset = torch.load(os.path.join(path, 'train'))
        valid_dataset = torch.load(os.path.join(path, 'valid'))
        test_dataset = torch.load(os.path.join(path, 'test'))
    else:
        os.system(f"mkdir {path}")
        train_dataset = ARDataset(args.key_size, args.value_size, sample_len=args.num_pairs, num_samples=args.train_size)
        valid_dataset = ARDataset(args.key_size, args.value_size, sample_len=args.num_pairs, num_samples=args.valid_size)
        test_dataset = ARDataset(args.key_size, args.value_size, sample_len=args.num_pairs, num_samples=args.test_size)

        torch.save(train_dataset, os.path.join(path, 'train'))
        torch.save(valid_dataset, os.path.join(path, 'valid'))
        torch.save(test_dataset,  os.path.join(path, 'test'))

    train_rnd_generator = torch.Generator()
    train_rnd_generator.manual_seed(args.seed)
    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size,  generator=train_rnd_generator,
                                  collate_fn=collate_fn, **kwargs)
    valid_dataloader = DataLoader(valid_dataset, batch_size=per_worker_batch_size,
                                  collate_fn=collate_fn, **kwargs)
    test_dataloader = DataLoader(test_dataset, batch_size=per_worker_batch_size,
                                  collate_fn=collate_fn, **kwargs)
    

    if args.valid_interval is None:
        args.valid_interval = args.log_interval

    # define model
    model_cls = get_cls_by_name(args.model_cls)

    logger.info(f'Using model class: {model_cls}')
    if not args.from_pretrained:
        model_cfg = AutoConfig.from_pretrained(args.model_cfg)
        model = model_cls(config=model_cfg)
    else:
        logger.info(f'Loading pretrained model: {args.from_pretrained}')
        model = model_cls.from_pretrained(args.from_pretrained)

    # ## add [GEN] token
    # model.resize_token_embeddings(len(tokenizer))
    
    ## load cpt of backbone model
    if args.backbone_cpt:
        backbone_cpt = os.path.join(args.backbone_cpt, "model_best.pth")
        cpt = torch.load(backbone_cpt, map_location='cpu')
        model.load_state_dict(cpt['model_state_dict'], strict=False)
        logger.info(f'Loaded baseline state dict from: {args.backbone_cpt}')

    # Pass memory settings to pretrained model
    if args.num_mem_tokens is not None:
        memory_cell_cls = get_cls_by_name(args.memory_cell_cls)
        recurrent_wrapper_cls = get_cls_by_name(args.recurrent_wrapper_cls)
        logger.info(f'Wrapping in: {memory_cell_cls} and {recurrent_wrapper_cls}')
        
        
        cell = memory_cell_cls(model, args.num_mem_tokens)
        model = recurrent_wrapper_cls(cell, 
                                      segment_size=block_size,
                                      max_n_segments=args.max_n_segments, 
                                    #   vary_n_segments=args.vary_n_segments,
                                      k2=args.k2,
                                      segment_alignment=args.segment_alignment
        )
                                    

        ## load cpt of rmt
        if args.model_cpt and args.model_cpt != 'None':
            model_cpt = os.path.join(args.model_cpt, "model_best/pytorch_model.bin")
            cpt = torch.load(model_cpt, map_location='cpu')
            model.load_state_dict(cpt, strict=False)
            logger.info(f'Loaded RMT state dict from: {args.model_cpt}')

    if args.freeze_model_weights:
        for n, p in model.named_parameters():
            # if 'memory' not in n and 'wte' not in n:
            if 'memory' not in n and 'lora' not in n:
                p.requires_grad = False
        logger.info(f'Frozen moodel weights')
        logger.info(f'Remaining parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}')

    # # fix the not-contiguous error with loralib and horovod
    # def make_contiguous(module):
    #     with torch.no_grad():
    #         for param in module.parameters():
    #             param.set_(param.contiguous())
    # make_contiguous(model)
    
    # define optimizer
    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

    logger.info(f'Using optimizer class: {optimizer_cls}')

    # todo: group optimizer params
    if optimizer_cls in [transformers.optimization.Adafactor, optimizers.Adafactor]:
        # https://github.com/huggingface/transformers/pull/9751/files -> transformers 4.3.0
        optimizer = optimizer_cls(model.parameters(), lr=args.lr,
                                  scale_parameter=args.scale_parameter,
                                  relative_step=args.relative_step,
                                  warmup_init=args.warmup_init,
                                  weight_decay=args.weight_decay)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # for encoder only classification
    def keep_for_metrics_fn(batch, output):
        # select data from batch and model output that would be used to compute metrics
        data = {}
        if 'generation_outputs' in output:
            data['labels'] = batch['labels']
            data['labels_mask'] = batch['labels_mask']

            data['generation_outputs'] = output['generation_outputs']
            # if 'labels_mask' in batch:
            #     data['generation_outputs'] = [data['generation_outputs'][i, mask] for i, mask in enumerate(batch['labels_mask'])]
        # if args.model_type == 'encoder':
            
            ##### booydar
            # data['predictions'] = torch.argmax(output['logits'].detach(), dim=-1)
        # data['labels'] = batch['labels']
        for key in batch.keys():
            if 'loss' in key: 
                data[key] = batch[key]
        # else:

        return data

    # HF datasets can compute metrics on each gpu process and then aggregate them on process with rank 0
    # synchronization is done by using temporay files on a shared filesystem
    # rank and number of workers is set by num_process and process_id params
    # BUT our Trainer aggregates all prediction from all gpus!
    #   this will lead to computing metrics for predictions repeated xN_GPUS times
    # need to try:
    # - keep_in_memory=True, may lead to OOM for large validation sets, after sync predictions and targets for the full
    #       validation set would be stored on each GPU -> xN_GPUs RAM
    #   - implemented currently
    # - compute metrics on batch lvl
    # - add support of HF metrics and turn off aggregation in case if metric has .add_batch method

    def metrics_fn(data):
        # compute metrics based on stored labels, predictions, ...
        metrics = {}
        y, p = None, None
        if 'generation_outputs' in data:
            y = data['labels']
            p = data['generation_outputs']

            metrics['exact_match'] = np.mean([(len(p_) >= args.value_size + 1) and torch.all(torch.tensor(y_)[-args.value_size - 1:] == torch.tensor(p_[-args.value_size - 1:])) \
                                              for p_, y_ in zip (p, y)])

            # replace -100 with pad token in labels
            # y = torch.stack([l[m] for l, m in zip(data['labels'], data['labels_mask'])])
            # y = data['labels'][:, -args.value_size - 1:-1]
            # p = data['generation_outputs']
            # if not hasattr(p, 'shape'):
            #     p = torch.stack([torch.tensor(x) for x in p])
            # # p = p[:, -args.value_size - 1:-1]

            # metrics['exact_match'] = np.mean([(len(y_) == len(p_)) and (y_ == p_) for p_, y_ in zip (p, y)])
            # metrics['exact_match'] = np.mean([y_ == p_ for p_, y_ in zip (p, y)])
            # preds = tokenizer.batch_decode(data['generation_outputs'], skip_special_tokens=False)
            # p = [p[:p.index(tokenizer.eos_token)] if tokenizer.eos_token in p else p for p in preds]
            if args.show_valid_examples > 0:
                for i in range(min(args.show_valid_examples, len(y))):
                    logger.info(f"labels: {data['labels'][i]}")
                    logger.info(f"gen: {data['generation_outputs'][i]}")
                    logger.info(f'y: {y[i][-args.value_size - 1:]}')
                    logger.info(f'p: {p[i][-args.value_size - 1:]}')
                    # logger.info(f'p ids: {data["generation_outputs"][i]}')
                    # logger.info('\n'.join([(y_, p_[:len(y_)], y_==p_[:len(y_)]) for p_, y_ in zip (p, y[:30])]))

                    logger.info('-' * 50)
            # todo: do we need to better clean P to remove tokens after eos? not remove special tokens only
        # elif args.model_type == 'encoder':
        #     y, p = data['labels'], data['predictions']

        # if y is not None and p is not None:
            # if args.model_type == 'encoder-decoder':
            # if not isinstance(y[0], list):
                # y = [[_y] for _y in y]
            # result = scrolls_metric.compute(predictions=p, references=y)
            # for metric_name in task_to_metric[args.task_name]:
            #     metrics[metric_name] = result[metric_name]

            # metrics['exact_match'] = np.mean([y_ == p_[:len(y_)] for p_, y_ in zip (p, y)])
            # elif args.model_type == 'encoder' and args.task_name == 'contract_nli':
            #     metrics['exact_match'] = accuracy_score(y, p) * 100
            #     metrics['f1_micro'] = f1_score(y, p, average='micro')
        return metrics

    # accelerate
    model, optimizer, train_dataloader, valid_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, None)

    ### booydar
    batch_metrics_fn = lambda _, y: {key: y[key] for key in y.keys() if (('loss' in key) or ('!log' in key))}
    trainer = Trainer(args, accelerator, model, optimizer, train_dataloader, valid_dataloader,
                      keep_for_metrics_fn=keep_for_metrics_fn, metrics_fn=metrics_fn,
                      ###booydar
                      batch_metrics_fn=batch_metrics_fn,
                    #   generate_kwargs={'max_new_tokens': int(args.value_size * 2)}
                      generate_kwargs={'pad_token_id': 102}
                      )

    # try:
    if not args.validate_only:
        # train loop
        trainer.train()
        # make sure all workers are done
        accelerator.wait_for_everyone()
        # run validation after training
        if args.save_best:
            best_model_path = str(Path(args.model_path) / 'model_best')
            logger.info(f'Loading best saved model from {best_model_path}')
            trainer.load(best_model_path)
        if valid_dataloader is not None:
            logger.info('Runnning validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False, split='valid')
        # if test_dataloader is not None:
        #     logger.info('Runnning validation on test data:')
            # trainer.validate(test_dataloader, write_tb=True, split='test')
        trainer.save_metrics(save_path=args.model_path)
    else:
        # run validation, do not write to tensorboard
        logger.info('Running validation on train set:')
        trainer.validate(train_dataloader, split='train', write_tb=True)
        if valid_dataloader is not None:
            logger.info('Running validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=True, split='valid')
        # if test_dataloader is not None:
        #     logger.info('Runnning validation on test data:')
        #     trainer.validate(test_dataloader, write_tb=True, split='test')
    # except Exception as e:
    #     print(f"Got exception: {e}")
    print('Done!')