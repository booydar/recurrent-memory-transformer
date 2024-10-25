import logging
from pathlib import Path
from itertools import chain
import os
import torch
import numpy as np
import random
import datasets
from torch.utils.data import DataLoader

from transformers import Trainer, TrainingArguments
from torch.nn.utils.rnn import pad_sequence

import accelerate
from peft import get_peft_model, LoraConfig, TaskType

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=logger_fmt, level=logging.INFO)
logger = logging.getLogger('')


# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

from transformers import AutoConfig, AutoTokenizer, HfArgumentParser  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model

from lm_experiments_tools.utils import get_cls_by_name, get_optimizer, prepare_run  # noqa: E402


parser = HfArgumentParser(TrainingArguments)
parser.add_argument('--task_name', type=str, help="Task name, wikitext, ...")
parser.add_argument('--tokenized_dataset', type=str, help="path to folder with tokenized hf dataset")
parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--show_valid_examples', type=int, default=0,
                    help='how many valid examples to show during training (default: 0)')
parser.add_argument('--sample_size', type=int, default=128, help='input sequnce length (default: 128).')
parser.add_argument('--val_sample_size', type=int, default=128, help='input sequnce length for validation (default: 128).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

parser.add_argument('--input_prefix', type=str, default='', help='add task prefix to an input string (default: "")')
parser.add_argument('--sliding_window', action='store_true', help='use slinding window attentinon mask, '
                    'eval on last segment only', default=False)

# model args
parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "")')
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: "")')
parser.add_argument('--model_cls', type=str, default='transformers:AutoModel',
                    help='model class name to use (default: transformers:AutoModel)')
parser.add_argument('--memory_cell_cls', type=str, default=None, help='cell class for RMT')
parser.add_argument('--recurrent_wrapper_cls', type=str, default=None, help='recurrent wrapper class for RMT')
parser.add_argument('--model_cpt', type=str, default=None, help='pretrained model checkpoint path')
parser.add_argument('--checkpoint', type=str, default=None, help='full experiment checkpoint')
parser.add_argument('--model_type', type=str, default='encoder-decoder',
                    help='model type, encoder, encoder-decoder, decoder, affects preprocessing '
                         '(default: encoder-decoder)')


# Aydar # RMT args
parser.add_argument('--segment_size', type=int, default=None, help='number of real tokens in block')
parser.add_argument('--num_mem_tokens', type=int, default=None, help='number of memory tokens.')
parser.add_argument('--max_n_segments', type=int, default=1, help='maximal segment number')
parser.add_argument('--vary_n_segments', action='store_true', default=False, help='Randomly choose segment number from 1 to max_n_segments')
parser.add_argument('--loss_from_last_seg_only', action='store_true', default=False, help='take loss from last segment only')
parser.add_argument('--no_loss_from_first_segment', action='store_true', default=False, help='turn off loss from first segment')

parser.add_argument('--min_sample_len', type=int, default=16000, help='min sample len in tokens')


parser.add_argument('--sum_loss', action='store_true', default=False,
                    help='with this flag task loss from all segments is summed')
parser.add_argument('--bptt_depth', type=int, default=-1, help='max number of previous segments in gradient computation.')
parser.add_argument('--segment_ordering', type=str, help='segment order', default='regular',
                    choices=['regular', 'reversed', 'bidirectional', 'repeat_first', 'last_memory_only'])
parser.add_argument('--retain_graph', action='store_true', help='Retain computation graph during backward pass', default=False)
parser.add_argument('--use_truncated_backward', action='store_true', default=False,
                    help='whether to use RMT truncated bptt method in backward')
parser.add_argument('--k1', type=int, default=-1, help='(not implemented) If not -1, gradient update is done each k1 segments')
parser.add_argument('--k2', type=int, default=-1, help='number of last segments used by backward')
parser.add_argument('--freeze_model_weights', action='store_true', default=False,
                    help='Stop training all model weights except memory layers')
parser.add_argument('--backbone_cpt', type=str, default=None, help='backbone model checkpoint path')


# tokenizer
parser.add_argument('--tokenizer', type=str, default=None, help='path or name of pre-trained HF Tokenizer')

# optimizer args
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer name: AdamW, Adafactor. (default: AdamW)')
parser.add_argument('--scale_parameter', action='store_true', default=False,
                    help='Adafactor scale_parameter (default: False)')
parser.add_argument('--relative_step', action='store_true', default=False,
                    help='Adafactor relative_step (default: False)')
parser.add_argument('--warmup_init', action='store_true', default=False,
                    help='Adafactor warmup_init (default: False)')

# LoRA args
parser.add_argument('--use_lora', action='store_true', default=False, help='')
parser.add_argument('--lora_attn_dim', type=int, default=8, help='')
parser.add_argument('--lora_attn_alpha', type=int, default=32, help='')
parser.add_argument('--lora_dropout', type=float, default=0.1, help='')


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

    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)

    # Prepare datasets
    logger.info(f'preparing dataset for {args.task_name}')

    with accelerator.main_process_first():
        if args.tokenized_dataset is not None:
            dataset = datasets.load_from_disk(args.tokenized_dataset)
        else:
            raise NotImplementedError("Implemented only for pre-tokenized datasets")

    segment_size = args.segment_size
    history_size = args.sample_size - segment_size

    if args.val_sample_size is not None:
        val_history_size = args.val_sample_size - segment_size
    else:
        val_history_size = history_size

    def group_texts(examples, segment_size, history_size=None):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if history_size is None:
            result = {
                k: [t[i : i + segment_size] for i in range(0, total_length, segment_size)]
                for k, t in concatenated_examples.items()
            }
        else:
            result = {
                k: [t[max({0, i - history_size}) : i + segment_size] for i in range(history_size, total_length, segment_size)]
                for k, t in concatenated_examples.items()
            }
        return result

    id_pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    def collate_fn(batch):
        input_ids = labels = [torch.tensor(b['tokens']) for b in batch]
        attention_mask = [torch.ones_like(b, dtype=int) for b in input_ids]


        labels_mask = [torch.ones_like(b, dtype=int) for b in input_ids]
        
        if getattr(args, 'loss_from_last_seg_only', False):
            for m in labels_mask:
                m[:-args.segment_size] = False

        if getattr(args, 'no_loss_from_first_segment', False):
            for m in labels_mask:
                m[:args.segment_size] = False

        input_ids = pad_sequence(input_ids, padding_value=id_pad_value, batch_first=True)
        labels = pad_sequence(labels, padding_value=-100, batch_first=True)
        attention_mask = pad_sequence(attention_mask, padding_value=0, batch_first=True)
        labels_mask = pad_sequence(labels_mask, padding_value=0, batch_first=True)

        collated = {'input_ids': input_ids,
                    'labels': labels, 
                    'attention_mask': attention_mask,
                    'labels_mask': labels_mask.bool()
                    }

        return collated

    def filter_by_len(sample, min_len=16000):
        return len(sample['tokens']) > min_len
    
    def filter_by_16k(sample):
        return len(sample['tokens']) > 16000
    
    if args.min_sample_len not in {16000, None}:
        train_dataset = dataset['train'].filter(lambda sample: filter_by_len(sample, args.min_sample_len))
    else:
        train_dataset = dataset['train'].filter(filter_by_16k)
    
    with accelerator.main_process_first():
        train_dataset = train_dataset.select_columns(['tokens']).map(lambda x: group_texts(x, segment_size, history_size),
                                                        batched=True, desc=f"Grouping train in chunks of {segment_size} and history {history_size}")
        valid_dataset = dataset["validation"].select_columns(['tokens']).map(lambda x: group_texts(x, segment_size, val_history_size), 
                                                             batched=True, desc=f"Grouping valid in chunks of {segment_size} and history {val_history_size}")
        test_dataset = dataset["test"].select_columns(['tokens']).map(lambda x: group_texts(x, segment_size, val_history_size), 
                                                             batched=True, desc=f"Grouping test in chunks of {segment_size} and history {val_history_size}")

    
    num_valid_examples = 100
    valid_inds = np.linspace(1, len(valid_dataset)-1, num_valid_examples).astype(int).tolist()
    valid_dataset = valid_dataset.select(valid_inds)

    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}

    # define model
    model_cls = get_cls_by_name(args.model_cls)

    logger.info(f'Using model class: {model_cls}')

    if not args.from_pretrained:
        model_cfg = AutoConfig.from_pretrained(args.model_cfg)
        model = model_cls(config=model_cfg)
    else:
        logger.info(f'Loading pretrained model: {args.from_pretrained}')
        model = model_cls.from_pretrained(args.from_pretrained)

    if args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_attn_dim, 
            lora_alpha=args.lora_attn_alpha, 
            lora_dropout=args.lora_dropout
            )
        model = get_peft_model(model, peft_config)
        logger.info(f'Added LoRA, trainable parameters with LoRA only:')
        model.print_trainable_parameters()
    

    ## load cpt of backbone model
    if args.backbone_cpt:
        cpt = torch.load(args.backbone_cpt, map_location='cpu')
        model.load_state_dict(cpt['model_state_dict'], strict=False)
        logger.info(f'Loaded baseline state dict from: {args.backbone_cpt}')

    # Pass memory settings to pretrained model
    if args.num_mem_tokens is not None:
        memory_cell_cls = get_cls_by_name(args.memory_cell_cls)
        recurrent_wrapper_cls = get_cls_by_name(args.recurrent_wrapper_cls)
        logger.info(f'Wrapping in: {memory_cell_cls} and {recurrent_wrapper_cls}')
        
        cell = memory_cell_cls(model, num_mem_tokens=args.num_mem_tokens)
        model = recurrent_wrapper_cls(cell, 
                                      segment_size=segment_size,
                                      max_n_segments=args.max_n_segments, 
                                      vary_n_segments=args.vary_n_segments,
                                      k2=args.k2,
        )
                                    

        ## load cpt of rmt
        if args.model_cpt and args.model_cpt != 'None':
            cpt = torch.load(args.model_cpt, map_location='cpu')
            model.load_state_dict(cpt, strict=False)
            logger.info(f'Loaded RMT state dict from: {args.model_cpt}')


    training_args_dict = {key: value for key, value in vars(args).items() if hasattr(TrainingArguments('.'), key)}

    training_args_dict['remove_unused_columns'] = False
    training_args_dict['save_safetensors'] = False
    training_args_dict['bf16'] = True
    training_args_dict['label_names'] = ['labels']

    training_args_dict['evaluation_strategy'] = 'steps'
    training_args_dict['per_device_eval_batch_size'] = training_args_dict.get('per_device_train_batch_size') // 2
    training_args_dict['eval_accumulation_steps'] = 32
    training_args_dict['gradient_checkpointing'] = True
    training_args_dict['gradient_checkpointing_kwargs'] = {'use_reentrant':False}
    training_args_dict['log_level'] = 'debug'
    
    training_args = TrainingArguments(**training_args_dict)
    # args.gradient_checkpointing = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        # test_dataset=test_dataset,
        # compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    print("Trainer Gradient Checkpointing Enabled:", trainer.args.gradient_checkpointing)
    
    trainer.evaluate()
    if not args.validate_only:
        trainer.train(resume_from_checkpoint=args.checkpoint)