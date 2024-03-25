import os
import json
import argparse

save_folder = "/home/jovyan/rmt/wip/base_models/gptconfigs/"
# default_config = {
#   "architectures": [
#     "GPTNeoXForCausalLM"
#   ],
#   "model_type": "gpt_neox",
#   "vocab_size": 128,
#   "hidden_size": 128, 
#   "num_hidden_layers": 1, 
#   "num_attention_heads": 1, 
#   "intermediate_size": 128, 
#   "max_position_embeddings": 2048,
#   "bos_token_id": 101,
#   "eos_token_id": 102,
#   "hidden_act": "gelu",
#   "rotary_pct": 0.25,
#   "rotary_emb_base": 10000,
#   "attention_dropout": 0.0,
#   "hidden_dropout": 0.0,
#   "classifier_dropout": 0.1,
#   "initializer_range": 0.02,
#   "layer_norm_eps": 1e-5,
#   "use_cache": True,
#   "tie_word_embeddings": False,
#   "use_parallel_residual": True
#   }


default_config = {
    "activation_function": "gelu_new",
    "architectures": [
      "GPT2LMHeadModel"
    ],
    "attn_pdrop": 0.1,
    "bos_token_id": 103,
    "embd_pdrop": 0.1,
    "eos_token_id": 102,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2",
    "n_ctx": 1024,
    "n_positions": 1024,
    "resid_pdrop": 0.1,
    "summary_activation": None,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": True,
    "summary_type": "cls_index",
    "summary_use_proj": True,
    "task_specific_params": {
      "text-generation": {
        "do_sample": False,
        "max_length": 50
      }
    },
    "vocab_size": 128
  }

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_size", default=128)
parser.add_argument("--num_hidden_layers", default=1)
parser.add_argument("--num_attention_heads", default=1)
args = parser.parse_args()

config = dict(**default_config)
config['n_embd'] = int(args.hidden_size)
config['n_layer'] = int(args.num_hidden_layers)
config['n_head'] = int(args.num_attention_heads)

config_name = f"gpt2_tiny_{args.num_hidden_layers}l{args.num_attention_heads}hd{args.hidden_size}"
print(f'Saving config {config_name}')
save_path = os.path.join(save_folder, f'{config_name}.json')
with open(save_path, 'w') as f:
    json.dump(config, f)