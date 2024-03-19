import numpy as np
import torch
import sys
import datasets
from transformers import AutoTokenizer
from babilong_utils import TaskDataset, SentenceSampler, NoiseInjectionDataset
from tqdm import tqdm

class QARetrievalEnv:
    def __init__(self,
                 sample,
                 retriever,
                 retr_tokenizer,
                 lm_tokenizer,
                 reward_model=None,
                 max_steps=3

    ):
        super().__init__()
        self.max_steps = max_steps
        self.max_sentence_len = 50
        self.max_batch_size = 64

        self.retriever = retriever
        self.retr_tokenizer = retr_tokenizer
        self.lm_tokenizer = lm_tokenizer
        self.rmodel = reward_model
        self.references = list(sample['references'])
        self.sentences = []

        background_text = self.lm_tokenizer.batch_decode(sample['background_text'])
        self.sentences.extend(background_text)

        self.sentences.extend(sample['facts'])
        self.sentences = np.array(self.sentences)
        # self.facts_ids = np.arange(len(self.sentences))

        self.question = sample['question']  # append as this is a single str
        self.sent_embeds = self.get_embeds(self.sentences)

        self.state = None
        self.available_acts = None
        self.chosen_sent_ids = None
        self.num_steps = 0


    def reset(self):
        if self.rmodel:
            self.rmodel.reset()
        self.num_steps = 0
        self.state = [self.question]
        self.available_acts = np.ones(len(self.sentences), dtype=bool)
        self.chosen_sent_ids = []
        return self._make_state()
    @property
    def device(self):
        return self.retriever.device

    @torch.no_grad()
    def get_embeds(self, sentences):
        batch = self.retr_tokenizer(list(sentences), padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.device)
        B = batch["input_ids"].shape[0]
        embeds = []
        for i in range(0, B, self.max_batch_size):
            subbatch = {k:v[i:i+self.max_batch_size] for k, v in batch.items()}
            embeds.append(self.retriever(**subbatch).to("cpu"))

        embeds = embeds[0] if len(embeds) == 1 else torch.cat(embeds, dim=0)
        return embeds

    def step(self, chosen_acts):
        all_acts = set(self.chosen_sent_ids +list(chosen_acts))
        self.chosen_sent_ids = sorted(all_acts)
        retrieved_sentences = self.sentences[self.chosen_sent_ids]
        self.state = [self.question] + list(retrieved_sentences)
        self.available_acts[self.chosen_sent_ids] = False
        self.num_steps += 1
        done = self.num_steps >= self.max_steps
        return self._make_state(), self._reward(), done

    def _make_state(self):
        s = [" ".join(s[:self.max_sentence_len] for s in self.state)]
        if len(s[0]) > 1024:
            print(f"State length is too big: L={len(s[0])}")

        state_embed = self.get_embeds(s)[0]

        return {
            "acts_embed": self.sent_embeds,
            "acts_text": self.sentences,
            "acts_mask": self.available_acts.copy(),
            "state_embed": state_embed,
        }

    def _reward(self):
        if not self.rmodel:
            return 0.

        return self.rmodel.reward(self)


class RetrievalPolicy:
    def act(self, state):
        raise NotImplementedError()


class RNDPolicy(RetrievalPolicy):
    def __init__(self, retrieve_k=1):
        super().__init__()
        self.retrieve_k = retrieve_k

    def act(self, state):
        action_mask = state['acts_mask']
        available_ids = action_mask.nonzero()[0]
        chosen_actions = np.random.choice(available_ids, size=self.retrieve_k, replace=False)
        return chosen_actions


class TopKPolicy(RetrievalPolicy):
    def __init__(self, retrieve_k=1):
        super().__init__()
        self.retrieve_k = retrieve_k

    def act(self, state):
        s_embed = state['state_embed']
        a_mask = state['acts_mask']
        a_embed = state['acts_embed'][a_mask]
        acts_ids = sorted(state['acts_mask'].nonzero()[0])
        scores = torch.inner(s_embed, a_embed)
        score_ids = torch.argsort(scores, descending=True)
        chosen_actions = [acts_ids[i] for i in score_ids[:self.retrieve_k]]
        return chosen_actions


class GroundTruthReward:
    def __init__(self):
        super().__init__()

    def reward(self, env : QARetrievalEnv, **kwargs):
        if env.num_steps < env.max_steps: return 0.

        is_retrieved = []
        for r in env.references:
            is_retrieved.append(r in env.state)

        all_retrieved = all(is_retrieved)
        return float(all_retrieved)

    def reset(self):
        pass

def evaluate(dataset, policy, retriever, retr_tokenizer, lm_tokenizer, max_steps=3):
    #policy = TopKPolicy(2)
    rewards = []
    N = len(dataset)
    for i in range(N):
        sample = dataset[i]
        env = QARetrievalEnv(
            sample, retriever, retr_tokenizer,
            lm_tokenizer, GroundTruthReward(), max_steps=max_steps
        )
        s = env.reset()
        done = False
        reward = None
        while True:
            if done: break
            actions = policy.act(s)
            s, reward, done = env.step(actions)

        rewards.append(reward)
        retrieval_acc = np.mean(rewards)
        print(f"\rit {i+1}/{N}, retrieval accuracy: {retrieval_acc:.3f}", end="")

        del env

    retrieval_acc = np.mean(rewards)
    print(f"FINAL retrieval accuracy: {retrieval_acc:.3f}")
    return retrieval_acc

def play(policy, sample):

    print(sample.keys())
    env = QARetrievalEnv(
        sample,
        retriever,
        retr_tokenizer,
        lm_tokenizer,
        reward_model=GroundTruthReward(),
        max_steps=3
    )
    s = env.reset()
    print(s.keys())
    print("Question:", env.question)
    print("Sentences:")
    for i, sent in enumerate(env.sentences):
        sent_visual = sent.replace('\n', ' ')
        print(f"#{i}: {sent_visual}")
    print("num actions:", len(s['acts_mask']))

    done = False
    reward = None
    print("\n################## START EPISODE ####################")
    while True:
        print(f"step#{env.num_steps}")
        print("action mask:\n", s['acts_mask'].astype(np.int64))
        print(f"state: {' '.join(env.state)}")
        print("reward:", reward)
        # print("state_embed:", s['state_embed'].shape)

        if done:
            print("DONE!")
            break

        actions = policy.act(s)
        print("selected actions:", actions)
        s, reward, done = env.step(actions)

    print("#########################################")


if __name__ == "__main__":


    task = "qa2_two-supporting-facts"

    train_path = f"data/tasks_1-20_v1-2/en-10k/{task}_train.txt"
    test_path = f"data/tasks_1-20_v1-2/en-10k/{task}_test.txt"
    noise_dataset_name = "pg19"
    noise_dataset = datasets.load_dataset(noise_dataset_name)

    task_dataset_train = TaskDataset(train_path,) #max_n_facts=10)
    task_dataset_test = TaskDataset(test_path,) #max_n_facts=10)

    # background text
    lm_tokenizer = AutoTokenizer.from_pretrained('gpt2')

    noise_sampler_train = SentenceSampler(noise_dataset['train'], tokenizer=lm_tokenizer)
    noise_sampler_test = SentenceSampler(noise_dataset['test'], tokenizer=lm_tokenizer)

    sample_size = 16000  # max number of tokens in sample
    dataset_train = NoiseInjectionDataset(task_dataset=task_dataset_train,
                                          noise_sampler=noise_sampler_train,
                                          tokenizer=lm_tokenizer,
                                          sample_size=sample_size)

    dataset_test = NoiseInjectionDataset(task_dataset=task_dataset_test,
                                         noise_sampler=noise_sampler_test,
                                         tokenizer=lm_tokenizer,
                                         sample_size=sample_size)




    contriever_path = "/home/griver/projects/ml/nlp/contriever"
    if contriever_path not in sys.path:
        sys.path.append(contriever_path)
    from src.contriever import Contriever
    from transformers import AutoTokenizer
    device = torch.device('cuda:0')
    retriever = Contriever.from_pretrained("facebook/contriever").to(device)
    retr_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")

    policy = TopKPolicy(1)
    evaluate(dataset_test, policy, retriever, retr_tokenizer, lm_tokenizer, max_steps=10)
    #play(policy, dataset_train[0])
    exit()
