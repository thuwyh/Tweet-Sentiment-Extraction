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
                   load_model, save_model, set_seed, write_event, evaluate, get_predicts_from_token_logits, map_to_word)


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
    args.vocab_path = args.bert_path# + 'vocab.txt'
    set_seed()

    run_root = Path('../experiments/' + args.run_root)
    DATA_ROOT = Path('../input/')
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_path, cache_dir=None, do_lower_case=True)
    args.tokenizer = tokenizer
    if args.bert_path.find('roberta'):
        collator = MyCollator()
    else:
        # this is for bert models
        collator = MyCollator(token_pad_value=0, type_pad_value=1)

    if args.mode in ['train', 'validate', 'validate5', 'validate55', 'teacherpred']:
        folds = pd.read_pickle(DATA_ROOT / args.train_file)
        train_fold = folds[folds['fold'] != args.fold]
        if args.no_neutral:
            train_fold = train_fold[train_fold['sentiment']!='neutral']
        valid_fold = folds[folds['fold'] == args.fold]
        # remove pseudo samples
        if 'type' in valid_fold.columns.tolist():
            valid_fold = valid_fold[valid_fold['type']=='normal']
        print(valid_fold.shape)
        print('training fold:', args.fold)
        if args.limit:
            train_fold = train_fold[:args.limit]
            valid_fold = valid_fold[:args.limit]

    if args.mode == 'train':
        if run_root.exists() and args.clean:
            shutil.rmtree(run_root)
        run_root.mkdir(exist_ok=True, parents=True)

        training_set = TrainDataset(train_fold, tokenizer=tokenizer)
        training_loader = DataLoader(training_set, collate_fn=collator,
                                     shuffle=True, batch_size=args.batch_size,
                                     num_workers=args.workers)

        valid_set = TrainDataset(valid_fold, tokenizer=tokenizer, mode='test')
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
                                  num_workers=args.workers)

        model = TweetModel(args.bert_path)
        model.cuda()

        config = RobertaConfig.from_pretrained(args.bert_path)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.lr, eps=args.adam_epsilon)
        if args.fp16:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level="O2", verbosity=0)

        total_steps = int(len(train_fold) * args.n_epochs / args.step / args.batch_size)
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=total_steps)

        if args.multi_gpu == 1:
            model = nn.DataParallel(model)

        train(args, model, optimizer, scheduler,
              train_loader=training_loader,
              valid_df=valid_fold,
              valid_loader=valid_loader, epoch_length=len(training_loader)*args.batch_size)

    elif args.mode == 'validate5':
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
        word_preds, scores = get_predicts_from_token_logits(all_start_preds, all_end_preds, valid_fold, args)
        metrics = evaluate(word_preds, valid_fold, args)
    
    elif args.mode == 'validate52':
        # 在答案层面融合
        valid_fold = pd.read_pickle(DATA_ROOT / args.local_test)
        config = AutoConfig.from_pretrained(args.bert_path)
        model = TweetModel(config=config)
        all_word_preds = []
        for fold in range(5):
            load_model(model, run_root / ('best-model-%d.pt' % fold))
            model.cuda()
            valid_set = TrainDataset(valid_fold, tokenizer=tokenizer, mode='test')
            valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
                                      num_workers=args.workers)
            all_senti_preds, fold_start_pred, fold_end_pred, fold_inst_preds = predict(
                model, valid_fold, valid_loader, args, progress=True)
            fold_word_preds, scores = get_predicts_from_token_logits(fold_start_pred, fold_end_pred, valid_fold, args)
            all_word_preds.append(fold_word_preds)

        word_preds = ensemble_words(all_word_preds)
        metrics = evaluate(word_preds, valid_fold, args)

    elif args.mode == 'validate':
        valid_fold = pd.read_pickle(DATA_ROOT / args.local_test)
        valid_set = TrainDataset(valid_fold, tokenizer=tokenizer, mode='test')
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
                                  num_workers=args.workers)
        valid_result = valid_fold.copy()
        config = AutoConfig.from_pretrained(args.bert_path)
        model = TweetModel(config=config)
        load_model(model, run_root / ('best-model-%d.pt' % args.fold))
        model.cuda()
        if args.multi_gpu == 1:
            model = nn.DataParallel(model)

        all_senti_preds, all_start_preds, all_end_preds, all_inst_preds = predict(
            model, valid_fold, valid_loader, args, progress=True)

        metrics = evaluate(
            all_senti_preds, all_start_preds, all_end_preds, all_inst_preds, valid_fold, args)

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
            preds = get_predicts_from_token_logits(all_senti_preds, all_start_preds, all_end_preds, test, args)
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
            preds = get_predicts_from_token_logits(None, all_start_preds, all_end_preds, test, args)

        test['selected_text'] = preds
        test[['textID','selected_text']].to_csv('submission.csv', index=False)

def train(args, model: nn.Module, optimizer, scheduler, *,
          train_loader, valid_df, valid_loader, epoch_length, n_epochs=None) -> bool:
    n_epochs = n_epochs or args.n_epochs

    run_root = Path('../experiments/' + args.run_root)
    model_path = run_root / ('model-%d.pt' % args.fold)
    best_model_path = run_root / ('best-model-%d.pt' % args.fold)
    best_model_loss_path = run_root / ('best-loss-%d.pt' % args.fold)
    
    best_valid_score = 0
    best_valid_loss = 1e10
    start_epoch = 0
    best_epoch = 0
    step = 0
    log = run_root.joinpath('train-%d.log' %
                            args.fold).open('at', encoding='utf8')

    update_progress_steps = int(epoch_length / args.batch_size / 100)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    # loss_fn = nn.NLLLoss()

    for epoch in range(start_epoch, start_epoch + n_epochs):
        model.train()
        tq = tqdm.tqdm(total=epoch_length)
        losses = []
        mean_loss = 0
        for i, (tokens, types, masks, targets, starts, ends, inst) in enumerate(train_loader):
            lr = get_learning_rate(optimizer)
            batch_size, length = tokens.size(0), tokens.size(1)
            masks = masks.cuda()
            tokens, targets = tokens.cuda(), targets.cuda()
            types = types.cuda()
            starts, ends, inst = starts.cuda(), ends.cuda(), inst.cuda()

            senti_out, start_out, end_out, inst_out = model(tokens, masks, types)
            start_out = start_out.masked_fill(~masks.bool(), -10000.0)
            end_out = end_out.masked_fill(~masks.bool(), -10000.0)
            # 正常loss
            start_loss = loss_fn(start_out, starts)
            end_loss = loss_fn(end_out, ends)
            inst_loss = loss_fn(inst_out.permute(0,2,1), inst)
            loss = (start_loss+end_loss)/2+inst_loss

            loss /= args.step

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if i%args.step==0:
                if args.max_grad_norm > 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            losses.append(loss.item() * args.step)
            mean_loss = np.mean(losses[-1000:])

            if i > 0 and i % update_progress_steps == 0:
                tq.set_description(f'Epoch {epoch}, lr {lr:.6f}')
                tq.update(update_progress_steps*args.batch_size)
                tq.set_postfix(loss=f'{mean_loss:.5f}')
            step += 1
        tq.close()
        valid_metrics = validation(model, valid_df, valid_loader, args)
        write_event(log, step, epoch=epoch, **valid_metrics)
        current_score = valid_metrics['dirty_score_word']
        save_model(model, str(model_path), current_score, epoch + 1)
        if current_score > best_valid_score:
            best_valid_score = current_score
            shutil.copy(str(model_path), str(best_model_path))
            best_epoch = epoch
            print('model saved')
    os.remove(model_path)
    return True


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
    word_preds, scores = get_predicts_from_token_logits(all_start_preds, all_end_preds, valid_df, args)
    metrics = evaluate(word_preds, valid_df, args)
    return metrics


if __name__ == '__main__':
    main()
