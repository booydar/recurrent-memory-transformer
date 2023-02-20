import json
import logging
import os
import re
import string
from pathlib import Path
import bisect
from prepro_char_based_targets import process_file

# from megatron.data.dataset_utils import get_indexed_dataset_

import horovod.torch as hvd
# from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import datasets

from lm_experiments_tools.trainer import Trainer, TrainerArgs

from modeling_roberta_bsln import RobertaHotpotQA, compute_metrics

# load_dotenv()

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# if CUDA_VISIBLE_DEVICES is not set make all gpus visible
if os.environ.get('CUDA_VISIBLE_DEVICES', None) is None:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(torch.cuda.device_count())])

logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
# first call to torch.cuda.device_count() sets visible gpus, following calls will not change the result
logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")

hvd.init()

import transformers  # noqa: E402
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser  # noqa: E402

from lm_experiments_tools.utils import collect_run_configuration, get_cls_by_name, get_optimizer  # noqa: E402
import lm_experiments_tools.optimizers as optimizers  # noqa: E402

# limit # of CPU threads to be used per pytorch worker, otherwise it might use all cpus and throttle gpus
# > 2 fails cause of https://github.com/pytorch/pytorch/issues/56615
# need to upgrade to torch>1.8.1
torch.set_num_threads(4)
# all gpus set with CUDA_VISIBLE_DEVICES are visible to process, indexing from 0 to ...
# torch.cuda.set_device(hvd.local_rank())

parser = HfArgumentParser(TrainerArgs)
parser.add_argument('--task_name', type=str, help='HotpotQA task name: "distractor", "fullwiki"')
parser.add_argument('--validate_only', action='store_true', default=False,
                    help='Skip training and run only validation. (default: False)')
parser.add_argument('--working_dir', type=str, default='.',
                    help='working dir, should be a dir with t5-experiments repo (default: .)')
parser.add_argument('--seed', type=int, default=42, help='random seed')

parser.add_argument('--input_seq_len', type=int, default=128, help='input sequnce length (default: 128).')
parser.add_argument('--target_seq_len', type=int, default=16, help='input sequnce length (default: 16).')
parser.add_argument('--data_n_workers', type=int, default=2, help='number of dataloader workers (default: 2)')

parser.add_argument('--source_prefix', type=str, default='', help='add task prefix to a source string (default: "")')

# model args
parser.add_argument('--from_pretrained', type=str, help='model name in HF Model Hub (default: "")')
parser.add_argument('--model_cfg', type=str, help='path to model configuration file (default: "")')
parser.add_argument('--model_cls', type=str, default='transformers:BertForPreTraining',
                    help='model class name to use (default: transformers:BertForPreTraining)')
parser.add_argument('--model_type', type=str, default='encoder',
                    help='model type, encoder, encoder-decoder, decoder, affects preprocessing (default: encoder)')

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


class HotpotDataset(Dataset):
    def __init__(self, data):
        self.data = process_file(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context = self.data[idx]['context']
        question = self.data[idx]['question']
        ans = self.data[idx]['answer']
        char_offsets = self.data[idx]['char_answer_offsets']
        return context, question, ans, char_offsets


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(s)))


if __name__ == '__main__':
    args = parser.parse_args()
    # set current working dir
    args.working_dir = str(Path(args.working_dir).expanduser().absolute())
    os.chdir(args.working_dir)
    if hvd.rank() == 0:
        logger.info(f'hvd size: {hvd.size()}')
        logger.info(f'FP16: {args.fp16}')

    if hvd.rank() == 0 and args.model_path is None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')

    # create model path and save configuration
    if hvd.rank() == 0 and args.model_path is not None:
        model_path = Path(args.model_path)
        if not model_path.exists():
            Path(model_path).mkdir(parents=True)
        args_dict = collect_run_configuration(args)
        # todo: if model path exists and there is config file, write new config file aside
        json.dump(args_dict, open(model_path/'config.json', 'w'), indent=4)
    if not args.from_pretrained:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained)
    tokenizer.add_tokens(['<t>', '</t>', '[/sent]', '[question]', '[/question]'], special_tokens=True)

    def ans_map_func(label):
        if label.lower() == 'no':
            return 0
        if label.lower() == 'yes':
            return 1
        return 2
    # collate_fn depends on model type (encoder, encoder-decoder)
    # collate_fn defines how data is prepared for the model
    if args.model_type == 'encoder':
        num_labels = 2
        encode_plus_kwargs = {'max_length': args.input_seq_len,
                              'truncation': 'only_second',
                              'padding': 'longest',
                              'pad_to_multiple_of': args.input_seq_len,
                              'return_overflowing_tokens': True,
                              'return_offsets_mapping': True}
        # encode_plus_kwargs = {'max_length': 512,
        #                       'truncation': True,
        #                       'padding': 'longest',
        #                       'pad_to_multiple_of': 64}
        def ans_map_func(label):
            if label.lower() == 'yes':
                return 0
            if label.lower() == 'no':
                return 1
            return 2

        def create_span_labels(char_offsets,cs):
            # offset_mapping = features.pop('offset_mapping')
            # sample_map = features['overflow_to_sample_mapping']
            # res_s = np.zeros(offset_mapping.shape[:2])
            # res_f = np.zeros(offset_mapping.shape[:2])
            res = {}
            bs = len(cs)
            offsets = [cxt['offset_mapping'] for cxt in cs]
            for seg in range(8):
                res[f'seg{seg}_starts'] = np.zeros((bs, cs[0]['offset_mapping'].shape[1]))
                res[f'seg{seg}_ends'] = np.zeros((bs, cs[0]['offset_mapping'].shape[1]))
  
            def _find_context(seq_ids, search_r=True):
                l_s = 1
                r_s = len(seq_ids)-2
                while r_s - l_s > 1:
                    m = (l_s+r_s) // 2
                    if seq_ids[m] == 1:
                        r_s = m
                    elif seq_ids[m] == 0:
                        l_s = m
                    else:
                        if seq_ids[m-2] == 0:
                            l_s = m
                        else:
                            r_s = m
                if not search_r:
                    return r_s, len(seq_ids)-2
                else:
                    l_f = r_s
                    r_f = len(seq_ids)-1
                    while r_f - l_f > 1:
                        m = (l_f+r_f) // 2
                        if seq_ids[m]:
                            l_f=m
                        else:
                            r_f = m
                    return r_s, l_f
            
            def _find_offsets(cont_s, cont_f, char_offsets, offset_mapping):
                ans_s, ans_f = [], []
                for char_offset in char_offsets:
                    if char_offset[0] < offset_mapping[cont_s][0] or char_offset[1] > offset_mapping[cont_f][1]:
                        continue
                    else:
                        l_s = cont_s
                        r_s = cont_f
                        while r_s - l_s > 1:
                            m = (l_s+r_s) // 2
                            if offset_mapping[m][0] <= char_offset[0]:
                                l_s = m
                            else:
                                r_s = m
                        l_f = l_s
                        r_f = cont_f
                        while r_f - l_f > 1:
                            m = (l_f+r_f) // 2
                            if offset_mapping[m][1] <= char_offset[1]:
                                l_f = m
                            else:
                                r_f = m
                        ans_s.append(l_s)
                        ans_f.append(r_f)
                return ans_s, ans_f

            for sample_id in range(bs):
                for i, offset in enumerate(cs[sample_id]['offset_mapping']):
                    if i >= 8:
                        break
                    cur_ans_offsets = char_offsets[sample_id]
                    cont_s, cont_f = _find_context(cs[sample_id].sequence_ids(i), search_r=bool(i == len(cs[sample_id]['offset_mapping'])-1))
                    start_pos, end_pos = _find_offsets(cont_s, cont_f, cur_ans_offsets, offset)
                    for j in range(len(start_pos)):
                        res[f'seg{i}_starts'][sample_id][start_pos[j]] = 1
                        res[f'seg{i}_ends'][sample_id][end_pos[j]] = 1
            res = {k:torch.from_numpy(v) for k,v in res.items()}
            return res, offsets
                

        def collate_fn(batch):
            context, question, ans, char_offsets = zip(*batch)
            bs = len(context)
            features = []
            for sample_id in range(bs):
                features.append(tokenizer.batch_encode_plus([(question[sample_id], context[sample_id])], return_tensors='pt', **encode_plus_kwargs))
            # map labels to ids
            ans_labels = np.array([ans_map_func(t) for t in ans])
            span_labels, offsets = create_span_labels(char_offsets, features)
            seg_map = torch.Tensor([[1]*min(8, len(features[sample_id]['overflow_to_sample_mapping'])) + [0]*max(0, 8-len(features[sample_id]['overflow_to_sample_mapping']))  for sample_id in range(bs)]).to(torch.int64)
            out_features  = {}
            for seg in range(8):
                out_features[f'seg{seg}_input_ids'] = []
                out_features[f'seg{seg}_attention_mask'] = []
            for seg in range(8):
                for sample_id in range(bs):
                    out_features[f'seg{seg}_input_ids'].append(features[sample_id]['input_ids'][seg,:].reshape(1, -1) if features[sample_id]['input_ids'].size(0) > seg else torch.ones(1,encode_plus_kwargs['max_length']))
                    out_features[f'seg{seg}_attention_mask'].append(features[sample_id]['attention_mask'][seg,:].reshape(1,-1) if features[sample_id]['attention_mask'].size(0) > seg else torch.zeros(1,encode_plus_kwargs['max_length']))
                out_features[f'seg{seg}_input_ids'] = torch.cat(out_features[f'seg{seg}_input_ids'], dim=0).to(torch.int64)
                out_features[f'seg{seg}_attention_mask'] = torch.cat(out_features[f'seg{seg}_attention_mask'], dim=0).to(torch.int64)
            return {'seg_map': seg_map,
                    'question_type_labels': torch.from_numpy(ans_labels),
                    'answers': ans,
                    'offsets': offsets,
                    'context': context,
                    **out_features, **span_labels}
    elif args.model_type == 'encoder-decoder':
        global_attention_first_token = False  # should be True for LED
        num_labels = 0
        encode_plus_kwargs = {'truncation': True,
                              'padding': 'longest',
                              'pad_to_multiple_of': 1}
        # generate predictions to fixed length
        generate_kwargs = {'max_length': args.target_seq_len, 'min_length': args.target_seq_len}
        # generate predictions to max targets length in batch
        generate_kwargs = {}

        def collate_fn(batch):
            inputs, labels = zip(*batch)
            if args.source_prefix:
                inputs = [args.source_prefix + inp for inp in inputs]
            features = tokenizer.batch_encode_plus(list(inputs), max_length=args.input_seq_len,
                                                   return_tensors='pt', **encode_plus_kwargs)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer.batch_encode_plus(list(labels), max_length=args.target_seq_len,
                                                     return_tensors='pt', **encode_plus_kwargs).input_ids
            labels[labels == tokenizer.pad_token_id] = -100
            features['labels'] = labels
            if 'global_attention_mask' in features:
                raise RuntimeError('What global attention mask for Longformer and LongformerEncoder-Decoder should be?')
            return features
    else:
        raise NotImplementedError('only encoder & encoder-decoder type of model is supported')

    # get train dataset
    if hvd.rank() == 0:
        logger.info(f'preparing dataset for: {args.task_name}')
    dataset = datasets.load_dataset('hotpot_qa', args.task_name)
    train_dataset = HotpotDataset(dataset['train'])
    # shuffle train data each epoch (one loop over train_dataset)
    train_sampler = DistributedSampler(train_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=True,
                                       drop_last=False, seed=args.seed)
    per_worker_batch_size = args.batch_size * args.gradient_accumulation_steps
    global_batch_size = per_worker_batch_size * hvd.size()
    kwargs = {'pin_memory': True, 'num_workers': args.data_n_workers}
    train_dataloader = DataLoader(train_dataset, batch_size=per_worker_batch_size, sampler=train_sampler,
                                  collate_fn=collate_fn, **kwargs)
    # get validation dataset
    valid_dataloader = None
    if hvd.rank() == 0:
        logger.info(f'preparing vallidation dataset for: {args.task_name}')
    valid_dataset = HotpotDataset(dataset['validation'])
    valid_sampler = DistributedSampler(valid_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=per_worker_batch_size, sampler=valid_sampler,
                                      collate_fn=collate_fn, **kwargs)
    if args.valid_interval is None:
        args.valid_interval = args.log_interval
    # else:
    #     valid_dataloader = None
    #     if hvd.rank() == 0:
    #         logger.info('No validation data is used.')
    # get test dataset
    # if args.test_data_path:
    #     if hvd.rank() == 0:
    #         logger.info(f'preparing test data from: {args.test_data_path}')
    #     test_data_path = Path(args.test_data_path).expanduser().absolute()
    #     test_dataset = HyperpartisanDataset(test_data_path)
    #     test_sampler = DistributedSampler(test_dataset, rank=hvd.rank(), num_replicas=hvd.size(), shuffle=False)
    #     test_dataloader = DataLoader(test_dataset, batch_size=per_worker_batch_size, sampler=test_sampler,
    #                                  collate_fn=collate_fn, **kwargs)

    # define model
    model_cls = get_cls_by_name(args.model_cls)
    if hvd.rank() == 0:
        logger.info(f'Using model class: {model_cls}')
    if not args.from_pretrained:
        model_cfg = AutoConfig.from_pretrained(args.model_cfg)
        model_cfg.num_labels = num_labels
        model = model_cls(config=model_cfg)
    else:
        if hvd.rank() == 0:
            logger.info(f'Loading pretrained model: {args.from_pretrained}')
        model = model_cls.from_pretrained(args.from_pretrained, paragraph_marker_token='<t>', sentence_marker_token=['/sent'], num_labels=num_labels)
    model.resize_token_embeddings(model.roberta.embeddings.word_embeddings.weight.shape[0]+5)
    # define optimizer
    optimizer_cls = get_optimizer(args.optimizer)
    if optimizer_cls is None:
        raise RuntimeError(f'{args.optimizer} was not found in optimizers, torch.optim, transformers.optimization')

    if hvd.rank() == 0:
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
        data['q_type_preds'] = torch.argmax(output['question_type_logits'], dim=-1).detach().cpu().numpy()
        data['q_type_labels'] = batch['question_type_labels']
        samples = []
        bs = batch['seg0_input_ids'].shape[0]
        seg_map = batch['seg_map']
        for sample_id in range(bs):
            cur_sample = {}
            cur_sample['start_logits'] = output['start_logits'].cpu().detach().numpy()[:seg_map[sample_id].sum(),sample_id,:]
            cur_sample['end_logits'] = output['end_logits'].cpu().detach().numpy()[:seg_map[sample_id].sum(),sample_id,:]
            cur_sample['offsets'] = batch['offsets'][sample_id].detach().cpu().numpy()
            cur_sample['context'] = batch['context'][sample_id]
            cur_sample['answers'] = batch['answers'][sample_id]
            samples.append(cur_sample)
        data['samples'] = samples
        # if args.model_type == 'encoder':
        #     data['labels'] = batch['labels']
        #     data['predictions'] = torch.argmax(output['logits'].detach(), dim=-1)
        # elif args.model_type == 'encoder-decoder' and 'generation_outputs' in output:
        #     # logger.info(f'{output["generation_outputs"].shape}')
        #     data['labels'] = batch['labels']
        #     data['generation_outputs'] = output['generation_outputs']
        return data

    # def metrics_fn(data):
    #     # compute metrics based on stored labels, predictions, ...
    #     metrics = {}
    #     y, p = None, None
    #     if args.model_type == 'encoder':
    #         y, p = data['labels'], data['predictions']
    #     elif args.model_type == 'encoder-decoder' and 'generation_outputs' in data:
    #         y = tokenizer.batch_decode(data['labels'], skip_special_tokens=True)
    #         p = tokenizer.batch_decode(data['generation_outputs'], skip_special_tokens=True)
    #         # if hvd.rank() == 0:
    #         #     logger.info(f'{y}')
    #         #     logger.info(f'{p}')
    #         #     for label, out in zip(data['labels'][:10], data['generation_outputs'][:10]):
    #         #         logger.info(f'{label} {out}')
    #         # map to labels
    #         y = [labels_map.get(normalize_answer(_y), 0) for _y in y]
    #         p = [labels_map.get(normalize_answer(_p), 0) for _p in p]
    #     if y is not None and p is not None:
    #         # accuracy, f1, precision, recall
    #         metrics['accuracy'] = accuracy_score(y, p)
    #         metrics['f1'] = f1_score(y, p)
    #         metrics['precision'] = precision_score(y, p)
    #         metrics['recall'] = recall_score(y, p)
    #     return metrics

    trainer = Trainer(args, model, optimizer, train_dataloader, valid_dataloader, train_sampler,
                      metrics_fn=compute_metrics, keep_for_metrics_fn=keep_for_metrics_fn,
                      generate_kwargs=generate_kwargs if args.use_generate_on_valid else {})

    if not args.validate_only:
        # train loop
        trainer.train()
        # make sure all workers are done
        hvd.barrier()
        # run validation after training
        if args.save_best:
            best_model_path = str(Path(args.model_path) / 'model_best.pth')
            if hvd.rank() == 0:
                logger.info(f'Loading best saved model from {best_model_path}')
            trainer.load(best_model_path)
        if valid_dataloader is not None:
            if hvd.rank() == 0:
                logger.info('Runnning validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False)
        # if args.test_data_path:
        #     if hvd.rank() == 0:
        #         logger.info('Runnning validation on test data:')
        #     trainer.validate(test_dataloader, split='test', write_tb=True)
    else:
        # run validation, do not write to tensorboard
        if hvd.rank() == 0:
            logger.info('Running validation on train set:')
        trainer.validate(train_dataloader, write_tb=False)
        if args.valid_data_path:
            if hvd.rank() == 0:
                logger.info('Running validation on valid data:')
            trainer.validate(valid_dataloader, write_tb=False)
        # if args.test_data_path:
        #     if hvd.rank() == 0:
        #         logger.info('Running validation on test data:')
        #     trainer.validate(test_dataloader, split='test', write_tb=False)
