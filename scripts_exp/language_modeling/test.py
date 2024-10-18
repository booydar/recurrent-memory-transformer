from datasets.utils.logging import get_verbosity
from datasets import load_dataset, load_from_disk
import datasets


dataset = load_from_disk("/home/jovyan/rmt/datasets/pg19/pg19_tokenized")

train_dataset = dataset['train'].select(range(100))
min_sample_len = 10001

def filter_by_len(sample, min_len=16000):
    return len(sample['tokens']) > min_len

def filter_by_16k(sample):
    return len(sample['tokens']) > 16000

if min_sample_len not in {16000, None}:
    train_dataset = train_dataset.filter(lambda sample: filter_by_len(sample, min_sample_len))
else:
    train_dataset = train_dataset.filter(filter_by_16k)


