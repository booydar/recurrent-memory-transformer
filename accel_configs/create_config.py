import os
import yaml
import json
import argparse


accel_config = {
'compute_environment': 'LOCAL_MACHINE',
'main_process_port': None,
'deepspeed_config':
    {'deepspeed_config_file': None,
    'zero3_init_flag': True},
'distributed_type': 'DEEPSPEED',
'downcast_bf16': 'no',
'machine_rank': 0,
'main_training_function': 'main',
'num_machines': 1,
'num_processes': None,
'rdzv_backend': 'static',
'same_network': True,
'tpu_env': [],
'tpu_use_cluster': False,
'tpu_use_sudo': False,
'use_cpu': False,
}

deepspeed_config = {
    "bf16": {
        "enabled": "auto"
    },

    "fp16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 2
    },
    "gradient_accumulation_steps": None,
    "gradient_clipping": 1.0,
    "train_batch_size": None,
    "train_micro_batch_size_per_gpu": None
}



parser = argparse.ArgumentParser()

parser.add_argument("--fp16", action='store_true', default=False)
parser.add_argument("--bf16", action='store_true', default=False)
parser.add_argument("--train_batch_size", default=256)
parser.add_argument("--train_micro_batch_size_per_gpu", default=256)
parser.add_argument("--gradient_accumulation_steps", default=1)
parser.add_argument("--np", default=1)
parser.add_argument("--gradient_clipping", default=1.0)

args = parser.parse_args()

if args.bf16:
    precision = "bf16_"
elif args.fp16:
    precision = "fp16_"
else:
    precision = ""

accel_config_path = "/home/jovyan/rmt/babilong/accel_configs/deepspeed_" + precision + "tbs{}bs{}g{}c{}np{}.yaml"
accel_config_path = accel_config_path.format(args.train_batch_size,
                                            args.train_micro_batch_size_per_gpu,
                                            args.gradient_accumulation_steps,
                                            args.gradient_clipping, 
                                            args.np)
deepspeed_config_path = "/home/jovyan/rmt/babilong/accel_configs/0s2_" + precision + "tbs{}bs{}g{}c{}.json"
deepspeed_config_path = deepspeed_config_path.format(args.train_batch_size,
                                                     args.train_micro_batch_size_per_gpu,
                                                     args.gradient_accumulation_steps,
                                                     args.gradient_clipping)

accel_config['num_processes'] = int(args.np)
accel_config['deepspeed_config']['deepspeed_config_file'] = deepspeed_config_path

deepspeed_config['fp16']['enabled'] = bool(args.fp16)
deepspeed_config['bf16']['enabled'] = bool(args.bf16)
deepspeed_config['train_batch_size'] = int(args.train_batch_size)
deepspeed_config['train_micro_batch_size_per_gpu'] = int(args.train_micro_batch_size_per_gpu)
deepspeed_config['gradient_accumulation_steps'] = int(args.gradient_accumulation_steps)
deepspeed_config['gradient_clipping'] = float(args.gradient_clipping)


print(f'Accelerate config {accel_config_path}')
with open(accel_config_path, 'w') as f:
    yaml.safe_dump(accel_config, f,  default_flow_style=False)

print(f'Deepspeed config {deepspeed_config_path}')
with open(deepspeed_config_path, 'w') as f:
    json.dump(deepspeed_config, f)
