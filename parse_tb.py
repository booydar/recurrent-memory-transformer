from pathlib import Path
import json
import pandas as pd

from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
# from tqdm.notebook import tqdm

TGT_COLS = ['task_name', 'from_pretrained', 'model_cfg', 'model_cls', 'model_type', 'lr', 'batch_size', 'HVD_SIZE', 'lr_scheduler', 'input_seq_len', 'max_n_segments', 'input_size', 'num_mem_tokens','segment_ordering','padding_side', 'model_path', 'sum_loss', 'inter_layer_memory', 'memory_layers', 'share_memory_layers','reconstruction_loss_coef', 'num_steps']

SILENT = True
def parse_tensorboard(path, scalars, silent=SILENT):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    # assert all(
    #     s in ea.Tags()["scalars"] for s in scalars
    # ),"" if silent else "some scalars were not found in the event accumulator"
    # return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}
    found_scalars = [s for s in scalars if s in ea.Tags()['scalars']]
    return {k: pd.DataFrame(ea.Scalars(k)) for k in found_scalars}


def parse_to_df(path, target_cols, metric_names, silent=SILENT):
    path = Path(path)

    logs = list(path.glob('**/*tfevents*'))

    experiments = []
    for p in logs:
    # for p in tqdm(logs):
        try:
            expr = json.load(open(p.parent / 'config.json', 'r'))
        except Exception as e:
            if not silent:
                print(f'error: {e}\n\tskip: {p}')
            continue
        metrics = {}
        try:
            metrics = parse_tensorboard(str(p), [f'{m}/iterations/valid' for m in metric_names])
        except Exception as e:
            if not silent:
                print(f'error: {e}\n\tskip: {p}')
        try:
            metrics_test = parse_tensorboard(str(p), [f'{m}/iterations/test' for m in metric_names])
        except Exception as e:
            metrics_test = {}
            if not silent:
                print(f'error: {e}\n\t no test metrics in: {p}')
        metrics.update(metrics_test)

        if len(metrics) == 0:
            continue
        for m in metric_names:
            if f'{m}/iterations/test' in metrics:
                expr[m] = metrics[f'{m}/iterations/test']['value'].item()
            if 'loss' in m or 'ppl' in m.lower() or 'bpc' in m.lower():
                expr[f'best_valid_{m}'] = metrics[f'{m}/iterations/valid']['value'].min()
            else:
                if f"{m}/iterations/valid" in metrics:
                    expr[f'best_valid_{m}'] = metrics[f'{m}/iterations/valid']['value'].max()
                else:
                    pass
                    # print(f"best_valid_{m} not found in metrics!\n{metrics.keys()}")

        # print(parse_tensorboard(str(p), ['loss/iterations/train'])['loss/iterations/train'].step)
        parsed = parse_tensorboard(str(p), ['loss/iterations/train'])
        if 'loss/iterations/train' in parsed:
            expr['num_steps'] = parsed['loss/iterations/train'].step.max()
        experiments += [expr]

    experiments = pd.DataFrame(experiments)
    # print('\n\ncolumns: ', experiments.columns)
    
    not_found_cols = [col for col in target_cols if col not in experiments.columns]
    if not_found_cols:
        if not silent:
            print(f'{not_found_cols} not found in columns!!\ncolumns:{experiments.columns}')
    
    found_cols = [col for col in target_cols if col in experiments.columns]
    experiments = experiments[found_cols]
    # print(experiments)
    return experiments
    # print('\n\ncolumns: ', experiments.columns)




# # babilong new 
# paths = [
#         '/home/bulatov/runs/babilong/',
#         ]

# paths = [Path(p) for p in paths]
# metric_names = ['exact_match']
# new_cols = ['input_size', 'k1', 'k2', 'freeze_model_weights', 'use_truncated_backward', 'retain_grad']#, 'noise_n_segments']
# target_cols = TGT_COLS + ['best_valid_exact_match', 'exact_match'] + new_cols
# out_path = 'results/babilong_new.csv'

# dfs = [parse_to_df(p, target_cols, metric_names) for p in paths]
# df = pd.concat(dfs)
# df.to_csv(out_path, index=False)
    
    
    
# # # HYP

# path = Path('/home/bulatov/bulatov/runs/finetune/debug/hyperpartisan_news_detection')
# metric_names = ['f1', 'precision', 'recall', 'accuracy']
# target_cols = ['f1', 'best_valid_f1', 'precision', 'best_valid_precision', 'recall', 'best_valid_recall', 'accuracy', 'best_valid_accuracy']
# out_path = 'results/hyp_new.csv'

# parse_to_csv(path, out_path, target_cols, metric_names)

# # # CNLI

# paths = ['/home/bulatov/bulatov/RMT_light/runs/contract_nli',
#         ]         
         
# paths = [Path(p) for p in paths]
# metric_names = ['exact_match', "loss"]
# target_cols = TGT_COLS + ['best_valid_exact_match', 'best_valid_loss']
# out_path = 'results/contract_nli_decoder.csv'

# dfs = [parse_to_df(p, target_cols, metric_names) for p in paths]
# df = pd.concat(dfs)
# df.to_csv(out_path, index=False)



# CNLI-archive
# paths = ['/home/bulatov/bulatov/RMT_light/runs/framework/contract_nli',
#          '/home/bulatov/bulatov/RMT_light/runs/test/contract_nli'
#         ]         
         
# paths = [Path(p) for p in paths]
# metric_names = ['exact_match']
# target_cols = TGT_COLS + ['best_valid_exact_match']
# out_path = 'results/contract_nli.csv'

# dfs = [parse_to_df(p, target_cols, metric_names) for p in paths]
# df = pd.concat(dfs)
# df.to_csv(out_path, index=False)


# CNLI -c curriculum


# paths = [
#         '/home/bulatov/bulatov/RMT_light/runs/curriculum/contract_nli',
#         ]         
         
# paths = [Path(p) for p in paths]
# metric_names = ['exact_match']
# target_cols = TGT_COLS + ['best_valid_exact_match']
# out_path = 'results/contract_nli_curriculum.csv'

# dfs = [parse_to_df(p, target_cols, metric_names) for p in paths]
# df = pd.concat(dfs)
# df.to_csv(out_path, index=False)



# # # QAsper

# paths = [
#         '/home/jovyan/rmt/runs/qasper/',
#         ]

# paths = [Path(p) for p in paths]
# metric_names = ['f1']
# new_cols = ['backbone_cpt', 'k2', 'model_cpt']
# target_cols = TGT_COLS + ['best_valid_f1'] + new_cols
# out_path = 'results/qasper_decoder.csv'

# dfs = [parse_to_df(p, target_cols, metric_names) for p in paths]
# df = pd.concat(dfs)
# df.to_csv(out_path, index=False)



# # quality

# paths = [
#         '/home/jovyan/rmt/runs/quality/',
#         ]

# paths = [Path(p) for p in paths]
# metric_names = ['exact_match']
# new_cols = ['backbone_cpt', 'k2', 'model_cpt']
# target_cols = TGT_COLS + ['best_valid_exact_match'] + new_cols
# out_path = 'results/quality_decoder.csv'

# dfs = [parse_to_df(p, target_cols, metric_names) for p in paths]
# df = pd.concat(dfs)
# df.to_csv(out_path, index=False)


# # Babi-long compare to curriculum

# paths = [
#         # '/home/bulatov/bulatov/RMT_light/runs/framework/babilong',
#         '/home/bulatov/bulatov/RMT_light/runs/compare_curriculum/babilong',
#         # '/home/bulatov/bulatov/RMT_light/runs/curriculum/babilong'
#         ]

# paths = [Path(p) for p in paths]
# metric_names = ['exact_match']
# target_cols = TGT_COLS + ['best_valid_exact_match']
# out_path = 'results/babilong_compare_curriculum.csv'

# dfs = [parse_to_df(p, target_cols, metric_names) for p in paths]
# df = pd.concat(dfs)
# df.to_csv(out_path, index=False)


# # Babi-long

# paths = [
#         # '/home/bulatov/bulatov/RMT_light/runs/framework/babilong',
#         '/home/bulatov/bulatov/RMT_light/runs/curriculum_task/babilong',
#         # '/home/bulatov/bulatov/RMT_light/runs/curriculum/babilong'
#         ]

# # path = Path('/home/bulatov/bulatov/RMT_light/runs/')
# paths = [Path(p) for p in paths]
# metric_names = ['exact_match']
# target_cols = TGT_COLS + ['best_valid_exact_match']
# out_path = 'results/babilong.csv'

# dfs = [parse_to_df(p, target_cols, metric_names) for p in paths]
# df = pd.concat(dfs)
# df.to_csv(out_path, index=False)


# # Babi-long random position

# paths = [
#         '/home/bulatov/bulatov/RMT_light/runs/curriculum_task/babilong_random',
#         ]

# # path = Path('/home/bulatov/bulatov/RMT_light/runs/')
# paths = [Path(p) for p in paths]
# metric_names = ['exact_match']
# target_cols = TGT_COLS + ['best_valid_exact_match']
# out_path = 'results/babilong_random.csv'

# dfs = [parse_to_df(p, target_cols, metric_names) for p in paths]
# df = pd.concat(dfs)
# df.to_csv(out_path, index=False)


# # Babi-long random position fix (no fact repetition in first seg)

# paths = [
#         '/home/bulatov/bulatov/RMT_light/runs/curriculum_task/babilong_random_v2',
#         ]

# # path = Path('/home/bulatov/bulatov/RMT_light/runs/')
# paths = [Path(p) for p in paths]
# metric_names = ['exact_match']
# target_cols = TGT_COLS + ['best_valid_exact_match']
# out_path = 'results/babilong_random_v2.csv'

# dfs = [parse_to_df(p, target_cols, metric_names) for p in paths]
# df = pd.concat(dfs)
# df.to_csv(out_path, index=False)


# # # Babi-long reasoning

# paths = [
#         '/home/bulatov/bulatov/RMT_light/runs/curriculum_task/babilong_reasoning',
#         ]

# # path = Path('/home/bulatov/bulatov/RMT_light/runs/')
# paths = [Path(p) for p in paths]
# metric_names = ['exact_match']
# target_cols = TGT_COLS + ['best_valid_exact_match']
# out_path = 'results/babilong_reasoning.csv'

# dfs = [parse_to_df(p, target_cols, metric_names) for p in paths]
# df = pd.concat(dfs)
# df.to_csv(out_path, index=False)

# # Associative retrieval

# paths = [
#         '/home/jovyan/rmt/runs/test/associative_retrieval',
#         ]

# paths = [Path(p) for p in paths]
# metric_names = ['exact_match']
# target_cols = TGT_COLS + ['best_valid_exact_match', 'key_size', 'value_size', 'num_pairs']
# out_path = 'results/ar.csv'

# dfs = [parse_to_df(p, target_cols, metric_names) for p in paths]
# df = pd.concat(dfs)
# df.to_csv(out_path, index=False)

    

paths = [
        '/home/jovyan/rmt/runs/babilong/',
        ]

paths = [Path(p) for p in paths]
metric_names = ['exact_match']
new_cols = ['input_size', 'k1', 'k2', 'freeze_model_weights', 'use_truncated_backward', 'retain_grad', 'task_name']#, 'noise_n_segments']
target_cols = TGT_COLS + ['exact_match'] + new_cols
out_path = 'results/babilong_p2.csv'

dfs = [parse_to_df(p, target_cols, metric_names) for p in paths]
df = pd.concat(dfs)
df.to_csv(out_path, index=False)
    

paths = [
        '/home/jovyan/rmt/runs/babilong_multitask/',
        ]

paths = [Path(p) for p in paths]
metric_names = ['exact_match']
new_cols = ['input_size', 'k1', 'k2', 'freeze_model_weights', 'use_truncated_backward', 'retain_grad', 'task_name']#, 'noise_n_segments']
target_cols = TGT_COLS + ['exact_match'] + new_cols
out_path = 'results/babilong_multitask.csv'

dfs = [parse_to_df(p, target_cols, metric_names) for p in paths]
df = pd.concat(dfs)
df.to_csv(out_path, index=False)
    
    

# # Associative retrieval

# paths = [
#         '/home/jovyan/rmt/runs/associative_retrieval_v3',
#         ]

# paths = [Path(p) for p in paths]
# metric_names = ['exact_match']
# target_cols = TGT_COLS + ['best_valid_exact_match', 'key_size', 'value_size', 'num_pairs']
# out_path = 'results/ar-v3.csv'

# dfs = [parse_to_df(p, target_cols, metric_names) for p in paths]
# df = pd.concat(dfs)
# df.to_csv(out_path, index=False)


# # #wikitext

# # paths = [
# #         '/home/jovyan/rmt/runs/',
# #         ]

# # # path = Path('/home/bulatov/bulatov/RMT_light/runs/')
# # paths = [Path(p) for p in paths]
# # metric_names = ['loss']
# # new_cols = ['model_cpt', 'backbone_cpt', 'k1', 'k2', 'freeze_model_weights', 'use_truncated_backward', 'retain_grad', 'model_cpt', 'vary_n_segments']
# # target_cols = TGT_COLS + ['best_valid_loss'] + new_cols
# # out_path = 'results/wikitext.csv'

# # dfs = [parse_to_df(p, target_cols, metric_names) for p in paths]
# # df = pd.concat(dfs)
# # df.to_csv(out_path, index=False)

# # pile 
# paths = [
#         '/home/jovyan/rmt/runs/pile/',
#         ]

# paths = [Path(p) for p in paths]
# metric_names = ['loss']
# new_cols = ['backbone_cpt', 'k1', 'k2', 'freeze_model_weights', 'use_truncated_backward', 'retain_grad']#, 'noise_n_segments']
# target_cols = TGT_COLS + ['best_valid_loss'] + new_cols
# out_path = 'results/pile.csv'

# dfs = [parse_to_df(p, target_cols, metric_names) for p in paths]
# df = pd.concat(dfs)
# df.to_csv(out_path, index=False)

# # arxiv 
# paths = [
#         '/home/jovyan/rmt/runs/arxiv/',
#         ]

# paths = [Path(p) for p in paths]
# metric_names = ['loss']
# new_cols = ['backbone_cpt', 'k1', 'k2', 'freeze_model_weights', 'use_truncated_backward', 'retain_grad']#, 'noise_n_segments']
# target_cols = TGT_COLS + ['best_valid_loss'] + new_cols
# out_path = 'results/arxiv.csv'

# dfs = [parse_to_df(p, target_cols, metric_names) for p in paths]
# df = pd.concat(dfs)
# df.to_csv(out_path, index=False)


# # memtest
# paths = [
#         '/home/jovyan/rmt/runs/memtest/arxiv/',
#         ]

# paths = [Path(p) for p in paths]
# metric_names = ['loss'] + [f'used_memory_gpu_{i}' for i in range(8)]
# new_cols = ['backbone_cpt', 'k1', 'k2', 'freeze_model_weights', 'use_truncated_backward', 'retain_grad']#, 'noise_n_segments']
# target_cols = TGT_COLS + ['time'] + new_cols
# out_path = 'results/memtest.csv'

# dfs = [parse_to_df(p, target_cols, metric_names) for p in paths]
# df = pd.concat(dfs)
# df.to_csv(out_path, index=False)