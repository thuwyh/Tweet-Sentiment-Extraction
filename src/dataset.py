import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, AutoConfig, AutoModel, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from utilsv4 import prepare, jaccard_list, decode
from nltk.corpus import wordnet
import pandas as pd
import numpy as np
import random

import warnings
warnings.filterwarnings("ignore")

def clean(x):
    sp_x = x.split()
    if len(sp_x[0]) == 1 and len(sp_x) > 1 and sp_x[0].lower() not in ['i', 'a', 'u'] and not sp_x[0].isdigit():
        return ' '.join(sp_x[1:])
    return x


class TrainDataset(Dataset):

    def __init__(self, data, tokenizer, mode='train', smooth=False, epsilon=0.15):
        super(TrainDataset, self).__init__()
        # self._tokens = data['tokens'].tolist()
        # self._sentilabel = data['senti_label'].tolist()
        # self._sentiment = data['sentiment'].tolist()
        # if 'type' in data.columns.tolist():
        #     self._type = data['type'].tolist()
        # else:
        #     self._type = ['normal']*len(self._tokens)

        self._tokenizer = tokenizer
        self._data = data

        self._data.dropna(subset=['text'], how='any', inplace=True)

        self._data['text'] = self._data['text'].apply(
            lambda x: ' '.join(x.lower().strip().split()))
        if 'selected_text' in self._data.columns.tolist():
            self._data['selected_text'] = self._data['selected_text'].apply(
                lambda x: ' '.join(x.lower().strip().split()))
            self._data['c_selected_text'] = self._data['selected_text'].apply(
                lambda x: clean(x))

        self._text = self._data['text'].tolist()
        self._sentiment = self._data['sentiment'].tolist()
        senti2label = {
            'positive':2,
            'negative':0,
            'neutral':1
        }
        self._data['senti_label']=self._data['sentiment'].apply(lambda x: senti2label[x])
        self._sentilabel = self._data['senti_label'].tolist()
        self.prepare_word()

        if mode == 'train':
            self._st = self._data['c_selected_text'].tolist()
            self.get_syns()
            self.get_label()
            self.evaluate_label()

        self._mode = mode
        self._smooth = smooth
        self._epsilon = epsilon
        # if mode in ['train', 'valid']:
        #     self._start = data['start'].tolist()
        #     self._end = data['end'].tolist()
        #     self._all_sentence = data['all_sentence'].tolist()
        #     self._inst = data['in_st'].tolist()
        # else:
        #     pass

        self._offset = 4 if isinstance(tokenizer, RobertaTokenizer) else 3

    def get_syns(self):
        syns_count = 0
        self._syns_map = {}
        for word in self._all_words:
            syn = wordnet.synsets(word)
            if len(syn) == 0:
                continue
            syn = syn[0].lemmas()[0].name()
            if syn != word:
                self._syns_map[word] = syn
                syns_count += 1
        print('total synonyms found:', syns_count)

    def prepare_word(self):
        data = []
        self._all_words = set()
        for text in self._text:
            words, first_char, invert_map = prepare(text)

            tokens, token_invert_map = [], []
            for idx, w in enumerate(words):
                self._all_words.add(w)
                # get tokens
                w = w.replace("`", "'")
                if first_char[idx]:
                    prefix = " "
                else:
                    prefix = ""
                for idx2, token in enumerate(self._tokenizer.tokenize(prefix+w)):
                    tokens.append(token)
                    token_invert_map.append(idx)
            data.append((words, first_char, tokens, token_invert_map, invert_map))
        words, first_char, tokens, invert_map, word_invert_map = zip(*data)
        self._words = words
        self._first_char = first_char
        self._invert_map = invert_map
        self._word_invert_map = word_invert_map
        self._tokens = tokens
        self._data['first_char'] = first_char
        self._data['words'] = words
        self._data['invert_map'] = invert_map # token id to word id
        self._data['word_invert_map'] = word_invert_map # word to pos in sentence
        print('total unique words:', len(self._all_words))

    def get_label(self):
        self._start_word_idx = []
        self._end_word_idx = []
        self._whole_sentence = []
        for idx in range(len(self._text)):
            text = self._text[idx]
            st = self._st[idx]
            words = self._words[idx]
            invert_map = self._word_invert_map[idx]

            temp = np.zeros(len(text))

            end_pos = 0
            start_pos = text.find(st, end_pos)
            first_end = start_pos+len(st)
            temp[start_pos:first_end] = 1
            
            label = []
            for word_idx, w in enumerate(words):
                if sum(temp[invert_map[word_idx]:invert_map[word_idx]+len(w)]) > 0:
                    label.append(word_idx)
            start_word_idx = min(label)
            end_word_idx = max(label)

            self._start_word_idx.append(start_word_idx)
            self._end_word_idx.append(end_word_idx)

            split_text = text.split()
            st = st.replace("`", "'")
            st_words = st.split()

            if len(st_words)/len(split_text) > 0.9:
                self._whole_sentence.append(1)
            else:
                self._whole_sentence.append(0)
        self._data['start'] = self._start_word_idx
        self._data['end'] = self._end_word_idx
        self._data['whole_sentence'] = self._whole_sentence

    def evaluate_label(self):
        def get_jaccard(x):
            label = x['selected_text'].split()
            words = x['words']
            start = x['start']
            end = x['end']
            label2 = decode(words, x['first_char'], start, end).split()
            return jaccard_list(label, label2)
        self._data['label_jaccard'] = self._data.apply(
            lambda x: get_jaccard(x), axis=1)
        print('label jaccard:', self._data['label_jaccard'].mean())

    def __len__(self):
        return len(self._tokens)

    def get_other_sample(self, idx, sentiment):
        while True:
            idx = random.randint(0, len(self._tokens)-1)
            if self._sentiment[idx] != sentiment:
                return self._tokens[idx]

    def __getitem__(self, idx):
        sentiment = self._sentiment[idx]
        if self._mode != 'train':
            # just return tokens and labels
            tokens = self._tokens[idx]
            start, end = 0, 0
            inst = [-100]*(len(tokens)+self._offset+1)
            whole_sentence = 0
        else:
            word_start, word_end = self._start_word_idx[idx], self._end_word_idx[idx]
            is_label, inst = [], []
            if random.random()<0.3:
                # aug, change the words
                words, origin_words = [], self._words[idx]
                first_char = self._first_char[idx]
                label = []
                tokens, token_invert_map = [], []
                
                for word_idx, w in enumerate(origin_words):
                    if random.random()<0.5: # and word_idx!=word_start and word_idx!=word_end:
                        if w in self._syns_map:
                            w = self._syns_map[w]
                    w = w.replace("`", "'")

                    prefix = " " if first_char[word_idx] else ""
                    for idx2, token in enumerate(self._tokenizer.tokenize(prefix+w)):
                        tokens.append(token)
                        token_invert_map.append(word_idx)
            else:
                tokens = self._tokens[idx]
                token_invert_map = self._invert_map[idx]
           
            for i in range(len(tokens)):
                if word_start<=token_invert_map[i]<=word_end:
                    is_label.append(i)
                    inst.append(1)
                else:
                    inst.append(0)

            start = min(is_label)+self._offset
            end = max(is_label)+self._offset
            inst = [-100]*self._offset+inst+[-100]
            whole_sentence = self._whole_sentence[idx]

        inputs = self._tokenizer.encode_plus(
            sentiment, tokens, return_tensors='pt')

        token_id = inputs['input_ids'][0]
        if 'token_type_ids' in inputs:
            type_id = inputs['token_type_ids'][0]
        else:
            type_id = torch.zeros_like(token_id)
        mask = inputs['attention_mask'][0]

        if self._mode=='train' and self._smooth:
            start_idx, end_idx = start, end
            start, end = torch.zeros_like(token_id, dtype=torch.float), torch.zeros_like(
                token_id, dtype=torch.float)
            if True:
                start[start_idx] += 1-self._epsilon
                end[end_idx] += 1-self._epsilon

                start += self._epsilon/len(mask)
                end += self._epsilon/len(mask)
            else:
                start[start_idx] += 1
                end[end_idx] += 1
            
        return token_id, type_id, mask, self._sentilabel[idx], start, end, torch.LongTensor(inst), whole_sentence


class MyCollator:

    def __init__(self, token_pad_value=1, type_pad_value=0):
        super().__init__()
        self.token_pad_value = token_pad_value
        self.type_pad_value = type_pad_value

    def __call__(self, batch):
        tokens, type_ids, masks, label, start, end, inst, all_sentence = zip(
            *batch)
        tokens = pad_sequence(tokens, batch_first=True,
                              padding_value=self.token_pad_value)
        type_ids = pad_sequence(
            type_ids, batch_first=True, padding_value=self.type_pad_value)
        masks = pad_sequence(masks, batch_first=True, padding_value=0)
        label = torch.LongTensor(label)
        if not isinstance(start[0], int):
            start = pad_sequence(start, batch_first=True, padding_value=0)
            end = pad_sequence(end, batch_first=True, padding_value=0)
        else:
            start = torch.LongTensor(start)
            end = torch.LongTensor(end)
        all_sentence = torch.FloatTensor(all_sentence)
        inst = pad_sequence(inst, batch_first=True, padding_value=-100)
        return tokens, type_ids, masks, label, start, end, inst, all_sentence


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(
        '../../bert_models/roberta_base/', cache_dir=None, do_lower_case=True)
    df = pd.read_csv('../input/tweet-sentiment-extraction/train_folds.csv')
    # df = df.iloc[:1600]
    dataset = TrainDataset(df, tokenizer, mode='train')
