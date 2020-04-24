import argparse
import json
import os
import random
import re
import shutil
from collections import OrderedDict, defaultdict
from functools import partial
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import tqdm
from apex import amp

from sklearn.model_selection import GroupKFold
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import (AdamW, get_cosine_schedule_with_warmup,
                                       get_linear_schedule_with_warmup)

from utilsv3 import (binary_focal_loss, get_learning_rate, jaccard_list, get_best_pred, ensemble, ensemble_words,
                   load_model, save_model, set_seed, write_event, evaluate, get_predicts, map_to_word)


class TrainDataset(Dataset):

    def __init__(self, data, tokenizer, mode='train', aug=False):
        super(TrainDataset, self).__init__()
        self._tokens = data['tokens'].tolist()
        self._sentilabel = data['senti_label'].tolist()
        self._sentiment = data['sentiment'].tolist()
        self._inst = data['in_st'].tolist()
        self._data = data
        self._mode = mode
        if mode in ['train', 'valid']:
            self._start = data['start'].tolist()
            self._end = data['end'].tolist()
        else:
            pass
        self._tokenizer = tokenizer
        self._offset = 4 if isinstance(tokenizer, RobertaTokenizer) else 3

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, idx):
        sentiment = self._sentiment[idx]
        
        inputs = self._tokenizer.encode_plus(
            sentiment, self._tokens[idx], return_tensors='pt')

        token_id = inputs['input_ids'][0]
        if 'token_type_ids' in inputs:
            type_id = inputs['token_type_ids'][0]
        else:
            type_id = torch.zeros_like(token_id)
        mask = inputs['attention_mask'][0]
        if self._mode == 'train':
            inst = [-100]*self._offset+self._inst[idx]+[-100]
            start = self._start[idx]+self._offset
            end = self._end[idx]+self._offset
        else:
            start, end = 0, 0
            inst = [-100]*len(token_id)
        return token_id, type_id, mask, self._sentilabel[idx], start, end, torch.LongTensor(inst)

class MyCollator:

    def __init__(self, token_pad_value=1, type_pad_value=0):
        super().__init__()
        self.token_pad_value = token_pad_value
        self.type_pad_value = type_pad_value

    def __call__(self, batch):
        tokens, type_ids, masks, label, start, end, inst = zip(*batch)
        tokens = pad_sequence(tokens, batch_first=True, padding_value=self.token_pad_value)
        type_ids = pad_sequence(type_ids, batch_first=True, padding_value=self.type_pad_value)
        masks = pad_sequence(masks, batch_first=True, padding_value=0)
        label = torch.LongTensor(label)
        start = torch.LongTensor(start)
        end = torch.LongTensor(end)
        inst = pad_sequence(inst, batch_first=True, padding_value=-100)
        return tokens, type_ids, masks, label, start, end, inst


class TweetModel(nn.Module):

    def __init__(self, pretrain_path=None, dropout=0.2, config=None):
        super(TweetModel, self).__init__()
        if config is not None:
            self.bert = AutoModel.from_config(config)
        else:
            self.bert = AutoModel.from_pretrained(
                pretrain_path, cache_dir=None)
        self.head = nn.Sequential(
            OrderedDict([
                ('clf', nn.Linear(self.bert.config.hidden_size, 3))
            ])
        )
        self.ext_head = nn.Sequential(
            OrderedDict([
                ('se', nn.Linear(self.bert.config.hidden_size, 2))
            ])
        )
        self.inst_head = nn.Linear(self.bert.config.hidden_size, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs, masks, token_type_ids=None, input_emb=None):
        seq_output, pooled_output = self.bert(
            inputs, masks, token_type_ids=token_type_ids, inputs_embeds=input_emb)
        out = self.head(pooled_output)
        se_out = self.ext_head(self.dropout(seq_output))  #()
        inst_out = self.inst_head(self.dropout(seq_output))
        return out, se_out[:, :, 0], se_out[:, :, 1], inst_out


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train', 'validate', 'predict', 'predict5',
                         'validate5', 'validate52'])
    arg('run_root')
    arg('--batch-size', type=int, default=16)
    arg('--step', type=int, default=1)
    arg('--workers', type=int, default=2)
    arg('--lr', type=float, default=0.00003)
    arg('--clean', action='store_true')
    arg('--n-epochs', type=int, default=3)
    arg('--limit', type=int)
    arg('--fold', type=int, default=0)
    arg('--multi-gpu', type=int, default=0)

    arg('--bert-path', type=str, default='../../bert_models/roberta_base/')
    arg('--train-file', type=str, default='train_roberta.pkl')
    arg('--local-test', type=str, default='localtest_roberta.pkl')
    arg('--test-file', type=str, default='test.csv')
    arg('--output-file', type=str, default='result.csv')
    arg('--no-neutral', action='store_true')

    arg('--epsilon', type=float, default=0.3)
    arg('--max-len', type=int, default=200)
    arg('--fp16', action='store_true')
    arg('--lr-layerdecay', type=float, default=1.0)
    arg('--max_grad_norm', type=float, default=-1.0)
    arg('--weight_decay', type=float, default=0.0)
    arg('--adam-epsilon', type=float, default=1e-8)
    arg('--offset', type=int, default=4)
    arg('--best-loss', action='store_true')
    arg('--post', action='store_true')
    arg('--temperature', type=float, default=1.0)

    args = parser.parse_args()
    args.vocab_path = args.bert_path
    set_seed()

    ensemble_models = [
        {
            'bert_path': '../../bert_models/bert_base_uncased/',
            'weight_path': '../experiments/test3_bert/',
            'model_type': 'bert',
            'test_file': '../input/localtest_bert.pkl'
        },
        {
            'bert_path': '../../bert_models/roberta_base/',
            'weight_path': '../experiments/test3_roberta3/',
            'model_type': 'roberta',
            'test_file': '../input/localtest_roberta.pkl'
        }
    ]

    run_root = Path('../experiments/' + args.run_root)
    DATA_ROOT = Path('../input/')
    
    

    if args.mode in ['train', 'validate', 'validate5', 'validate55', 'teacherpred']:
        folds = pd.read_pickle(DATA_ROOT / args.train_file)
        train_fold = folds[folds['fold'] != args.fold]
        if args.no_neutral:
            train_fold = train_fold[train_fold['sentiment']!='neutral']
        valid_fold = folds[folds['fold'] == args.fold]
        print(valid_fold.shape)
        print('training fold:', args.fold)
        if args.limit:
            train_fold = train_fold[:args.limit]
            valid_fold = valid_fold[:args.limit]

    if args.mode == 'validate5':
        valid_fold = pd.read_pickle(DATA_ROOT / args.local_test)
        config = AutoConfig.from_pretrained(args.bert_path)
        model = TweetModel(config=config)
        all_start_preds, all_end_preds = [], []
        for fold in range(5):
            load_model(model, run_root / ('best-model-%d.pt' % fold))
            model.cuda()
            valid_set = TrainDataset(valid_fold, tokenizer=tokenizer, mode='test')
            valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
                                      num_workers=args.workers)
            all_senti_preds, fold_start_pred, fold_end_pred, fold_inst_preds = predict(
                model, valid_fold, valid_loader, args, progress=True)
            all_start_preds.append(fold_start_pred)
            all_end_preds.append(fold_end_pred)
        all_start_preds, all_end_preds = ensemble(None, all_start_preds, all_end_preds, valid_fold)
        word_preds = get_predicts(all_start_preds, all_end_preds, valid_fold, args)
        metrics = evaluate(word_preds, valid_fold, args)
    
    elif args.mode == 'validate52':
        # 在答案层面融合
        all_word_preds = []
        for m in ensemble_models:
            valid_fold = pd.read_pickle(m['test_file'])
            valid_fold.sort_values(by='textID', inplace=True)
            tokenizer = AutoTokenizer.from_pretrained(m['bert_path'], cache_dir=None, do_lower_case=True)
            args.tokenizer = tokenizer
            config = AutoConfig.from_pretrained(m['bert_path'])
            model = TweetModel(config=config)
            if m['model_type']=='roberta':
                collator = MyCollator()
                args.offset = 4
            else:
                # this is for bert models
                collator = MyCollator(token_pad_value=0, type_pad_value=1)
                args.offset = 3
        
            for fold in range(5):
                load_model(model, m['weight_path'] + ('best-model-%d.pt' % fold))
                model.cuda()
                valid_set = TrainDataset(valid_fold, tokenizer=tokenizer, mode='test')
                valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
                                        num_workers=args.workers)
                all_senti_preds, fold_start_pred, fold_end_pred, fold_inst_preds = predict(
                    model, valid_fold, valid_loader, args, progress=True)
                fold_word_preds = get_predicts(fold_start_pred, fold_end_pred, valid_fold, args)
                all_word_preds.append(fold_word_preds)
                print(fold_word_preds[0:5])

        word_preds = ensemble_words(all_word_preds)
        metrics = evaluate(word_preds, valid_fold, args)

    elif args.mode in ['predict', 'predict5']:
        test = pd.read_csv(DATA_ROOT / 'tweet-sentiment-extraction'/args.test_file)
        if args.limit:
            test = test.iloc[:args.limit]
        data = []
        for text in test['text'].tolist():
            split_text = text.split()
            tokens, invert_map, first_token = [], [], []
            for idx, w in enumerate(split_text):
                for idx2, token in enumerate(tokenizer.tokenize(' '+w)):
                    tokens.append(token)
                    invert_map.append(idx)
                    first_token.append(True if idx2==0 else False)
            data.append((tokens, invert_map, first_token))
        tokens, invert_map, first_token = zip(*data)
        test['tokens'] = tokens
        test['invert_map'] = invert_map
        test['first_token'] = first_token
        senti2label = {
            'positive':2,
            'negative':0,
            'neutral':1
        }
        test['senti_label']=test['sentiment'].apply(lambda x: senti2label[x])

        test_set = TrainDataset(test, vocab_path=args.vocab_path, mode='test', max_len=args.max_len)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                 num_workers=args.workers)
        config = AutoConfig.from_pretrained(args.bert_path)
        model = TweetModel(config=config)
        
        if args.mode == 'predict':
            load_model(model, run_root / ('best-model-%d.pt' % args.fold))

            model.cuda()
            model = amp.initialize(model, None, opt_level="O2", verbosity=0)
            if args.multi_gpu == 1:
                model = nn.DataParallel(model)
            all_senti_preds, all_start_preds, all_end_preds = predict(
                model, test, test_loader, args, progress=True)
            preds = get_predicts(all_senti_preds, all_start_preds, all_end_preds, test, args)
        if args.mode == 'predict5':
            all_start_preds, all_end_preds = [], []
            for fold in range(5):
                load_model(model, run_root / ('best-model-%d.pt' % fold))
                model.cuda()
                _, fold_start_preds, fold_end_preds = predict(model, test, test_loader, args, progress=True, for_ensemble=True)
                fold_start_preds = map_to_word(fold_start_preds, test, args)
                fold_end_preds = map_to_word(fold_end_preds, test, args)
                all_start_preds.append(fold_start_preds)
                all_end_preds.append(fold_end_preds)
            all_start_preds, all_end_preds = ensemble(None, all_start_preds, all_end_preds, test)
            preds = get_predicts(None, all_start_preds, all_end_preds, test, args)

        test['selected_text'] = preds
        test[['textID','selected_text']].to_csv('submission.csv', index=False)


def predict(model: nn.Module, valid_df, valid_loader, args, progress=False, for_ensemble=False) -> Dict[str, float]:
    # run_root = Path('../experiments/' + args.run_root)
    model.eval()
    all_end_pred, all_senti_pred, all_start_pred, all_inst_out = [], [], [], []
    if progress:
        tq = tqdm.tqdm(total=len(valid_df))
    with torch.no_grad():
        for tokens, types, masks, _, _, _, _ in valid_loader:
            if progress:
                batch_size = tokens.size(0)
                tq.update(batch_size)
            masks = masks.cuda()
            tokens = tokens.cuda()
            types = types.cuda()
            senti_out, start_out, end_out, inst_out = model(tokens, masks, types)
            start_out = start_out.masked_fill(~masks.bool(), -1000)
            end_out= end_out.masked_fill(~masks.bool(), -1000)
            all_senti_pred.append(np.argmax(senti_out.cpu().numpy(), axis=-1))
            inst_out = torch.softmax(inst_out, dim=-1)
            for idx in range(len(start_out)):
                all_start_pred.append(start_out[idx,:].cpu())
                all_end_pred.append(end_out[idx,:].cpu())
                all_inst_out.append(inst_out[idx,:,1].cpu())
            assert all_start_pred[-1].dim()==1

    all_senti_pred = np.concatenate(all_senti_pred)
    
    if progress:
        tq.close()
    return all_senti_pred, all_start_pred, all_end_pred, all_inst_out


def validation(model: nn.Module, valid_df, valid_loader, args, save_result=False, progress=False):
    run_root = Path('../experiments/' + args.run_root)

    all_senti_preds, all_start_preds, all_end_preds, all_inst_out = predict(
        model, valid_df, valid_loader, args)
    word_preds = get_predicts(all_start_preds, all_end_preds, valid_df, args)
    metrics = evaluate(word_preds, valid_df, args)
    return metrics


if __name__ == '__main__':
    main()
