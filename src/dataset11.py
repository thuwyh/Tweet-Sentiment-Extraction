import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, AutoConfig, AutoModel, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from nltk.corpus import wordnet
import pandas as pd
import numpy as np
import random
import re

import warnings
warnings.filterwarnings("ignore")


def clean(x):
    sp_x = x.split()
    if len(sp_x[0]) == 1 and len(sp_x) > 1 and sp_x[0].lower() not in ['i', 'a', 'u'] and not sp_x[0].isdigit():
        return ' '.join(sp_x[1:])
    return x


def broken(x, y):
    if y > 0 and x[y-1] not in [' ', '.', ',', '!', '?']:
        return 1
    return 0


def get_pos(x, y):
    return x.find(y)


def broken_start(x, y):

    if y > 0 and x[y-1].isalpha():
        return True
    return False


def broken_end(x, y):
    if y < len(x) and x[y] != ' ':
        return True
    return False


def get_clean_label(x):
    shift = x['shift']
    if shift < 1 or x['start_pos_clean'] == 0:
        return x['selected_text']

    # 不修复shift=1不断头的
    if shift==1 and  not x['broken_start']:
        return x['selected_text']

    text = x['text']
    start = x['start_pos_origin']
    end = x['end_pos_origin']

#     while(len(text[start+shift-1:end+shift-1].strip()) == 0):
#         shift += 1
    if shift==1:
        new_st = text[start+shift:end].strip()
    else:
        # 对于shift>1的，都应该修复，除非修复之后还是断头
        parts = x['selected_text'].split()
        
        if len(parts)==1 or len(parts[0])>shift:
            return x['selected_text']
        else:
            new_st = text[start+shift:end+shift-1].strip()
    assert len(new_st)>0
    return new_st


def replace_punc(x):
    return x.replace("'", "\"").replace("`", "'")


def get_extra_space_count(x):
    prev_space = True
    space_counts = []
    count = 0
    for c in x:
        if c == ' ':
            if prev_space:
                count += 1
            space_counts.append(count)
            prev_space = True
        else:
            space_counts.append(count)
            prev_space = False
    return space_counts


    

class TrainDataset(Dataset):

    def __init__(self, data, old_data, tokenizer, mode='train', smooth=False, epsilon=0.0, distill=False, offset=4):
        super(TrainDataset, self).__init__()

        self._tokenizer = tokenizer
        self._data = data

        self._data.dropna(subset=['text'], how='any', inplace=True)

        self._sentiment2 = data['raw_sentiment'].tolist()

        self._data['text'] = self._data['text'].str.rstrip()
        self._data['clean_text'] = self._data['text'].apply(
            lambda x: ' '.join(x.strip().split()))

        self._data['train_text'] = self._data['text'].apply(
            lambda x: replace_punc(x))
        self._data['extra_space'] = self._data['text'].apply(
            lambda x: get_extra_space_count(x))

        if 'selected_text' in self._data.columns.tolist():

            self._data['selected_text'] = self._data['selected_text'].str.rstrip()

            self._data['clean_st'] = self._data['selected_text'].apply(
                lambda x: ' '.join(x.strip().split()))
            self._data['start_pos_origin'] = self._data.apply(
                lambda x: get_pos(x['text'], x['selected_text']), axis=1)
            self._data['end_pos_origin'] = self._data['start_pos_origin'] + \
                self._data['selected_text'].str.len()
            self._data['start_pos_clean'] = self._data.apply(
                lambda x: get_pos(x['clean_text'], x['clean_st']), axis=1)
            self._data['end_pos_clean'] = self._data['start_pos_clean'] + \
                self._data['clean_st'].str.len()

            self._data['broken_start'] = self._data.apply(
                lambda x: broken_start(x['clean_text'], x['start_pos_clean']), axis=1)
            self._data['broken_end'] = self._data.apply(
                lambda x: broken_end(x['clean_text'], x['end_pos_clean']), axis=1)

            self._data['shift'] = self._data.apply(
                lambda x: x['extra_space'][x['end_pos_origin']-1], axis=1)
            self._data['to_end'] = self._data['end_pos_origin'] >= self._data['text'].str.len()

            self._data['new_st'] = self._data.apply(
                lambda x: get_clean_label(x), axis=1)

            self._data['whole_sentence'] = self._data.apply(lambda x: len(
                x['selected_text'])/len(x['text']) > 0.95, axis=1).astype(int)
            self._whole_sentence = self._data['whole_sentence'].tolist()

        self._text = self._data['text'].tolist()
        self._sentiment = self._data['sentiment'].tolist()

        senti2label = {
            'positive': 2,
            'negative': 0,
            'neutral': 1
        }
        self._data['senti_label'] = self._data['sentiment'].apply(
            lambda x: senti2label[x])
        self._sentilabel = self._data['senti_label'].tolist()

        self.prepare_words()

        if mode == 'train':
            self._st = self._data['new_st'].tolist()
            self.get_label()

        self._mode = mode
        self._smooth = smooth
        self._epsilon = epsilon
        self._distill = distill

        if distill:
            self._start_pred = data['start_pred'].tolist()
            self._end_pred = data['end_pred'].tolist()
        self._offset = offset

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

    def get_token_and_label(self, idx):
        # for synonym augmentation
        text = self._text[idx]
        words = self._words[idx]
        offsets = self._offsets[idx]
        st = self._st[idx]

        temp = np.zeros(len(text))

        start_pos = text.find(st)
        end_pos = start_pos+len(st)
        temp[start_pos:end_pos] = 1

        tokens = []
        labels = []
        for word_idx, word in enumerate(words):

            if sum(temp[offsets[word_idx][0]:offsets[word_idx][1]])>0:
                in_selected_text = True
            else:
                in_selected_text = False

            # augmentation
            if random.random() < 0.5:  # and word_idx!=word_start and word_idx!=word_end:
                if word in self._syns_map:
                    word = self._syns_map[word]

            if word_idx==0 or words[word_idx-1]==' ':
                prefix = ' '
            else:
                prefix = ''
            if word==' ':
                tokens.append("Ġ")
                if in_selected_text:
                    labels.append(len(tokens))
            else:
                for t in self._tokenizer.tokenize(prefix+word):
                    if random.random()<0.02:
                        random_token_id = random.randint(4, self._tokenizer.vocab_size)
                        t = self._tokenizer.convert_ids_to_tokens(random_token_id)
                    tokens.append(t)
                    if in_selected_text:
                        labels.append(len(tokens))
        start_token_idx = min(labels)
        end_token_idx = max(labels)
        return tokens, start_token_idx, end_token_idx

    def prepare_words(self):
        all_offsets = []
        all_words = []
        all_tokens = []
        all_invert_maps = []
        self._all_words = set()
        for text in self._text:
            prev_punc = True
            words = []
            offset = []
            tokens = []
            invert_map = []
            for idx, c in enumerate(text):
                if c in [' ','.',',','!','?','(',')',';',':','-','=',"/","<","`"]:
                    prev_punc = True
                    words.append(c)
                    offset.append(idx)
                else:
                    if prev_punc:
                        words.append(c)
                        offset.append(idx)
                        prev_punc = False
                    else:
                        words[-1]+=c
            offset = [(x, x+len(y)) for x, y in zip(offset, words)]
            for word_idx, word in enumerate(words):
                self._all_words.add(word)
                if word_idx==0 or words[word_idx-1]==' ':
                    prefix = ' '
                else:
                    prefix = ''
                if word==' ':
                    tokens.append("Ġ")
                    invert_map.append(word_idx)
                else:
                    for t in self._tokenizer.tokenize(prefix+word):
                        tokens.append(t)
                        invert_map.append(word_idx)
            all_words.append(words)
            all_offsets.append(offset)
            all_tokens.append(tokens)
            all_invert_maps.append(invert_map)
        
        self._offsets = all_offsets
        self._words = all_words
        self._tokens = all_tokens
        self._invert_map = all_invert_maps
        
        self._data['offsets'] = all_offsets
        self._data['words'] = all_words
        self._data['invert_map'] = all_invert_maps
        # print('total unique words:', len(self._all_words))
        print('mean token length', sum(map(len, self._tokens))/len(self._tokens))
        

    def get_label(self):
        self._start_token_idx = []
        self._end_token_idx = []

        for idx in range(len(self._text)):
            text = self._text[idx]
            st = self._st[idx].strip()
            offset = self._offsets[idx]
            token = self._tokens[idx]
            word = self._words[idx]
            invert_map = self._invert_map[idx]
            temp = np.zeros(len(text))

            end_pos = 0
            start_pos = text.find(st, end_pos)
            first_end = start_pos+len(st)
            temp[start_pos:first_end] = 1

            if start_pos < 0:
                print(text)
                print(st)

            label = []
            for token_idx, t in enumerate(token):
                word_idx = invert_map[token_idx]
                start = offset[word_idx][0]
                end = offset[word_idx][1]
                if sum(temp[start:end])>0:
                    label.append(token_idx)
            if len(label) == 0:
                print(start_pos, first_end)
                print(text)
                print(st)
                # print(offset)
                # print(token)
                # print(invert_map)
            start_token_idx = min(label)
            end_token_idx = max(label)

            self._start_token_idx.append(start_token_idx)
            self._end_token_idx.append(end_token_idx)

        self._data['start'] = self._start_token_idx
        self._data['end'] = self._end_token_idx


    def __len__(self):
        return len(self._text)

    def __getitem__(self, idx):
        text = self._text[idx]
        sentiment = self._sentiment[idx]
        sentiment2 = self._sentiment2[idx]
        if self._mode != 'train':
            # just return tokens and labels
            tokens = self._tokens[idx]
            start_idx, end_idx = 0, 0
            inst = [-100]*(len(tokens)+self._offset+1)
            start = end = inst
            whole_sentence = 0
        else:
            inst = []
            # if random.random()<0.1:
            #     tokens, token_start, token_end = self.get_token_and_label(idx)
            # else:
            token_start, token_end = self._start_token_idx[idx], self._end_token_idx[idx]
            tokens = self._tokens[idx]

            for i in range(len(tokens)):
                if token_start <= i <= token_end:
                    inst.append(1)
                else:
                    inst.append(0)

            start_idx = token_start+self._offset
            end_idx = token_end+self._offset
            inst = [-100]*self._offset+inst+[-100]
            whole_sentence = self._whole_sentence[idx]

        inputs = self._tokenizer.encode_plus(
            sentiment+'</s></s> '+sentiment2, tokens, return_tensors='pt') #+sentiment2

        token_id = inputs['input_ids'][0]
        if 'token_type_ids' in inputs:
            type_id = inputs['token_type_ids'][0]
        else:
            type_id = torch.zeros_like(token_id)
        mask = inputs['attention_mask'][0]

        if self._distill:
            start = torch.FloatTensor([0]*self._offset+list(self._start_pred[idx][:len(tokens)])+[0])
            end = torch.FloatTensor([0]*self._offset+list(self._end_pred[idx][:len(tokens)])+[0])
            assert len(start) == len(token_id)
        else:
            epsilon = 0 #0.1/len(token_id)#*len(mask)*random.random()
            start = torch.rand_like(token_id, dtype=torch.float)
            end = torch.rand_like(token_id, dtype=torch.float)

            start = start*epsilon/torch.sum(start)
            end = end*epsilon/torch.sum(end)

            start[start_idx] += 1-epsilon
            end[end_idx] += 1-epsilon
        return token_id, type_id, mask, self._sentilabel[idx], start, end, start_idx, end_idx, torch.LongTensor(inst), whole_sentence


class MyCollator:

    def __init__(self, token_pad_value=1, type_pad_value=0):
        super().__init__()
        self.token_pad_value = token_pad_value
        self.type_pad_value = type_pad_value

    def __call__(self, batch):
        tokens, type_ids, masks, label, start, end, start_idx, end_idx, inst, all_sentence = zip(
            *batch)
        tokens = pad_sequence(tokens, batch_first=True,
                              padding_value=self.token_pad_value)
        type_ids = pad_sequence(
            type_ids, batch_first=True, padding_value=self.type_pad_value)
        masks = pad_sequence(masks, batch_first=True, padding_value=0)
        label = torch.LongTensor(label)
        start = pad_sequence(start, batch_first=True, padding_value=0)
        end = pad_sequence(end, batch_first=True, padding_value=0)

        start_idx = torch.LongTensor(start_idx)
        end_idx = torch.LongTensor(end_idx)
        all_sentence = torch.LongTensor(all_sentence)
        inst = pad_sequence(inst, batch_first=True, padding_value=-100)
        return tokens, type_ids, masks, label, start, end, start_idx, end_idx, inst, all_sentence


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(
        '../../bert_models/roberta_base/', cache_dir=None, do_lower_case=True)
    df = pd.read_csv('../input/tweet-sentiment-extraction/train_folds.csv')
    # df = df.iloc[:1600]
    dataset = TrainDataset(df, tokenizer, mode='train')
