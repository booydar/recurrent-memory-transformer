from datasets import load_dataset
import numpy as np
from transformers import AutoModel, AutoTokenizer

dataset = load_dataset('pg19')

model_path = "/home/jovyan/kuratov/models/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_path)

def tokenize(sample):
    sample['tokens'] = tokenizer.encode(sample['text'], return_tensors='pt')[0]
    return sample

new_ds = dataset.map(tokenize, batch_size=256)
new_ds.save_to_disk('pg19_tokenized')