from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
# import os
# os.environ['HF_HOME'] = "/home/jovyan/.cache/huggingface/transformers"

from datasets.utils.logging import get_verbosity
import datasets
# Get and print the cache directory
cache_dir = datasets.config.HF_DATASETS_CACHE
print(f"Cache directory: {cache_dir}")

    # dataset = load_from_disk("/home/jovyan/rmt/datasets/pg19/pg19_tokenized")
    # def filter_by_len(sample, min_len=16000):
    #     return len(sample['tokens']) > min_len
    # filtered = dataset['train'].filter(filter_by_len)