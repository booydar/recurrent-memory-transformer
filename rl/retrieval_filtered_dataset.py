import torch
from rl.retrieval_env import QARetrievalEnv, RetrievalPolicy, TopKPolicy
from babilong_utils import NoiseInjectionDataset, sum_lengths
from torch.utils.data import Dataset
import sys
contriever_path = "/home/griver/projects/ml/nlp/contriever"
if contriever_path not in sys.path:
    sys.path.append(contriever_path)
from src.contriever import Contriever
from transformers import AutoTokenizer


# class NoiseInjectionDatasetNoTokenization(NoiseInjectionDataset):
#
#
#     def __getitem__(self, ind):
#         pass


class SequentialRetrievalPostprocessor(Dataset):
    def __init__(
        self,
        dataset: NoiseInjectionDataset,
        device: str="cuda",
        top_k :int=1,
        num_retrieval_steps: int=10,
        output_sample_size: int=460,
    ):
        self.base_dataset = dataset
        self.policy = TopKPolicy(top_k)
        self.device = torch.device(device)
        self.num_retrieval_steps = num_retrieval_steps
        self.output_sample_size = output_sample_size
        self.retriever = Contriever.from_pretrained("facebook/contriever").to(device)
        self.retr_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

    def __getitem__(self, ind):
        sample = self.base_dataset.__getitem__(ind)
        retr_sentences = self.multi_step_retrieval(sample)
        # pretrained model i'm using now doesn't scale to ANY other length
        # and also don't want to show the same performance
        # if i change rmt evaluation even slightly (╯°□°）╯︵ ┻━┻
        # as result of these complications we need to add noisy sentences AGAIN
        # to pad sample length
        retrieval_sample = self.add_noise_samples(sample, retr_sentences)
        return retrieval_sample


    def __len__(self):
        return len(self.base_dataset)

    def multi_step_retrieval(self, sample):
        env = QARetrievalEnv(
            sample, self.retriever,
            self.retr_tokenizer,
            self.base_dataset.tokenizer,
            max_steps=self.num_retrieval_steps
        )

        s = env.reset()
        while True:
            actions = self.policy.act(s)
            s, reward, done = env.step(actions)
            if done:
                break
        return list(env.state)

    def add_noise_samples(self, base_sample, retr_sentences):
        sample = dict(base_sample)
        lm_tokenizer = self.base_dataset.tokenizer
        noise_sampler = self.base_dataset.noise_sampler
        facts_tok = lm_tokenizer(retr_sentences[1:])['input_ids']

        sample_size = self.output_sample_size
        task_len = sum_lengths(facts_tok)
        # print(f'sum length facts len: {task_len}')
        background_text_len = sample_size - task_len
        # print(f"background len: {background_text_len}")
        background_text = noise_sampler.get_sample(background_text_len)
        sample['background_text'] = background_text
        possible_positions = range(len(background_text) + 1)
        fact_positions = self.base_dataset.gen.choice(possible_positions, len(facts_tok))
        fact_positions.sort()
        sample['fact_positions'] = fact_positions  # positions of facts between noise sentences

        updated_sample = [[] for _ in range(len(background_text) + 1)]
        for fact, pos in zip(facts_tok, fact_positions):
            updated_sample[pos].append(fact)

        for i, s in enumerate(background_text):
            updated_sample[i].append(s)

        flat = [i for s in updated_sample for i in s]
        tokens = [i for s in flat for i in s]

        sample['input_tokens'] = tokens
        return sample
