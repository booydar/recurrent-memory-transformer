import random
from tqdm import tqdm
# import spacy
import json as json
from collections import Counter
import numpy as np
import os.path
import argparse
import torch
# import pickle
import torch
import os
from joblib import Parallel, delayed

import torch

# nlp = spacy.blank("en")

import bisect
import re

#from transformers import BertTokenizer
#model_name = 'bert-base-uncased'
#tokenizer = BertTokenizer.from_pretrained(model_name)


def find_nearest(a, target, test_func=lambda x: True):
    idx = bisect.bisect_left(a, target)
    if (0 <= idx < len(a)) and a[idx] == target:
        return target, 0
    elif idx == 0:
        return a[0], abs(a[0] - target)
    elif idx == len(a):
        return a[-1], abs(a[-1] - target)
    else:
        d1 = abs(a[idx] - target) if test_func(a[idx]) else 1e200
        d2 = abs(a[idx-1] - target) if test_func(a[idx-1]) else 1e200
        if d1 > d2:
            return a[idx-1], d2
        else:
            return a[idx], d1

def fix_span(para, span):
    span = span.strip()
    parastr = "".join(para)
    assert span in parastr, '{}\t{}'.format(span, parastr)
    best_indices = []

    if span == parastr:
        return (0, len(parastr))

    for m in re.finditer(re.escape(span), parastr):
        begin_offset, end_offset = m.span()

        best_indices.append([begin_offset, end_offset])

    assert len(best_indices) != 0
    return best_indices

def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        token = token.strip()# for tokens with right and left spaces that not separated by 2 spaces in the text
        pre = current
        current = text.find(token, current)
        if current < 0:
            print(f"{token} not found in {text}")
            raise Exception(f"{token} not found in\n{text}\ntokens = {tokens}\npre {pre}")
        spans.append((current, current + len(token)))
        current += len(token)
    return spans

def prepro_sent(sent):
    return sent#.lower()
    # return sent.replace("''", '" ').replace("``", '" ')

def _process_article(article, with_special_seps=False):
    question_start_sep = ''
    question_end_sep = ''
    title_start_sep = ''
    title_end_sep = ''
    sent_end_sep = ''
    if with_special_seps:
        #[question]q q q[/question]<t>title 1</t>s_11 s_12 s_13[/sent]s_21 s_22 s_23[/sent]...<t>title 2</t>s_21 s_22 s_23[/sent]....[/par]yes no null 

        question_start_sep = '[question]'
        question_end_sep = '[/question]'
        title_start_sep = '<t>' # indicating the start of the title of a paragraph (also used for loss over paragraphs)
        title_end_sep = '</t>'
        sent_end_sep = '[/sent]' # indicating the end of the title of a sentence (used for loss over sentences)
    
    para_titles = article['context']['title']
    para_sentences = article['context']['sentences']
    paragraphs = list(zip(para_titles, para_sentences))
    # some articles in the fullwiki dev/test sets have zero paragraphs
    if len(paragraphs) == 0:
        paragraphs = [['some random title', 'some random stuff']]

    text_context = ''
    
    
    def _process(sent, is_sup_fact, is_title=False):
        nonlocal text_context
        
        if is_title:
            #sent = ' {}. '.format(sent)
            sent = title_start_sep + sent + title_end_sep
        else:
            sent += sent_end_sep
            
        
        text_context += sent
        
        
    ques_txt = question_start_sep + article['question'] + question_end_sep
    
    supp_sent_ids = []
    supp_sent_texts = []
    supp_sent_char_offsets = []
    supp_title_ids = []
    supp_title_texts = []
    supp_title_char_offsets = []
    if 'supporting_facts' in article:
        supp_title_texts = article['supporting_facts']['title']
        supp_title_ids = [para_titles.index(item) for item in supp_title_texts]
        sp_set = set(list(zip(*[supp_title_texts, article['supporting_facts']['sent_id']]))) #set(list(map(tuple, article['supporting_facts'])))
    else:
        sp_set = set()

    sent_counter = 0
    for para in paragraphs:
        cur_title, cur_para = para[0], para[1]
        _process(prepro_sent(cur_title), False, is_title=True)
        if cur_title in supp_title_texts:
            if with_special_seps:
                supp_title_token_start = text_context.rfind(title_start_sep)
                supp_title_char_offsets.append([supp_title_token_start, supp_title_token_start + len(title_start_sep)])
        
        for sent_id, sent in enumerate(cur_para):
            
            is_sup_fact = (cur_title, sent_id) in sp_set
            _process(prepro_sent(sent), is_sup_fact)
            if is_sup_fact:
                supp_sent_ids.append(sent_counter)
                supp_sent_texts.append(sent)
                if with_special_seps:
                    supp_sent_token_start = text_context.rfind(sent_end_sep)
                    supp_sent_char_offsets.append([supp_sent_token_start, supp_sent_token_start + len(sent_end_sep)])
            sent_counter += 1
            
    if 'answer' in article:
        answer = article['answer'].strip()
        if answer.lower() == 'yes':
            best_indices = [[-1, -1]]
        elif answer.lower() == 'no':
            best_indices = [[-2, -2]]
        else:
            if article['answer'].strip() not in ''.join(text_context):
                # in the fullwiki setting, the answer might not have been retrieved
                # use (0, 1) so that we can proceed
                best_indices = [[0, 0]]
            else:
                best_indices = fix_span(text_context, article['answer'].strip())
                
                
    else:
        # some random stuff
        answer = 'random'
        best_indices = (0, 1)
    
    example = {'context': text_context,
               'question': ques_txt,
               'answer': answer,
               'char_answer_offsets': best_indices, 
               'id': article['id'],
               'supp_title_char_offsets': supp_title_char_offsets,
               'supp_sent_char_offsets': supp_sent_char_offsets,
               'supp_title_ids': supp_title_ids,
               'supp_title_texts': supp_title_texts,
               'supp_sent_ids': supp_sent_ids,
               'supp_sent_texts': supp_sent_texts,
              }
    
    return example

def process_file(data, with_special_seps=False):
    
    examples = []
    eval_examples = {}

    outputs = Parallel(n_jobs=12, verbose=10)(delayed(_process_article)(article, with_special_seps) for article in data)
    print("{} questions in total".format(len(outputs)))

    return outputs