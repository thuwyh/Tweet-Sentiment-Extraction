import argparse
import json
import shutil
import os
from pathlib import Path
from typing import Dict
from functools import partial

import numpy as np
import pandas as pd
import torch
import tqdm
import re

from torch import nn
from torch.utils.data import DataLoader
from apex import amp
from sklearn.model_selection import GroupKFold

from utils import (
    binary_focal_loss, save_model, jaccard_list,
    get_learning_rate, set_seed,
    write_event, load_model)

from transformers import BertConfig, BertTokenizer, BertModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss, confusion_matrix
from torch.utils.data import Dataset, WeightedRandomSampler
import random
from torch.nn.utils.rnn import pad_sequence
from collections import OrderedDict, defaultdict




class TrainDataset(Dataset):

    def __init__(self, data, vocab_path=None, do_lower=True, mode='train', aug=False, max_len=200):
        super(TrainDataset, self).__init__()
        self._tokens = data['tokens'].tolist()
        self._sentilabel = data['senti_label'].tolist()
        # self._qid = data['qid'].tolist()
        self._data = data
        self._mode = mode
        if mode in ['train', 'valid']:
            self._start = data['start'].tolist()
            self._end = data['end'].tolist()
        else:
            pass
        self._max_len = max_len
        self._tokenizer = BertTokenizer.from_pretrained(
            vocab_path, cache_dir=None, do_lower_case=do_lower)

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, idx):
        inputs = self._tokenizer.encode_plus(self._tokens[idx], return_tensors='pt')
        toksn_id = inputs['input_ids'][0]
        # type_id = inputs['token_type_ids'][0]
        if self._mode=='train':
            start = self._start[idx]+1
            end = self._end[idx]+1
        else:
            start, end = 0, 0
        return toksn_id, self._sentilabel[idx], start, end
        

def collate_fn(batch):
    tokens, label, start, end = zip(*batch)
    tokens = pad_sequence(tokens, batch_first=True)
    label = torch.LongTensor(label)
    start = torch.LongTensor(start)
    end = torch.LongTensor(end)
    return tokens, label, start, end


class TweetModel(nn.Module):

    def __init__(self, pretrain_path=None, dropout=0.2, config=None):
        super(TweetModel, self).__init__()
        if config is not None:
            self.bert = BertModel(config)
        else:
            self.bert = BertModel.from_pretrained(
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

    def forward(self, inputs, masks, token_type_ids=None, input_emb=None):
        seq_output, pooled_output = self.bert(
            inputs, masks, token_type_ids=token_type_ids, inputs_embeds=input_emb)
        out = self.head(pooled_output)
        se_out = self.ext_head(seq_output+pooled_output.unsqueeze(1))
        return out, se_out[:,:,0], se_out[:,:,1]


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train', 'validate', 'predict', 'validate5','validate55','teacherpred'])
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

    arg('--bert-path', type=str, default='../../bert_models/large_uncased_wwm/')
    arg('--train-file', type=str, default='train.pkl')
    arg('--test-file', type=str, default='dev.csv')
    arg('--output-file', type=str, default='result.csv')

    arg('--epsilon', type=float, default=0.3)

    arg('--max-len', type=int, default=200)

    arg('--lr-layerdecay', type=float, default=1.0)
    arg('--max_grad_norm', type=float, default=-1.0)
    arg('--adam-epsilon', type=float, default=1e-8)

    arg('--best-loss', action='store_true')
    arg('--temperature', type=float, default=1.0)

    args = parser.parse_args()
    args.vocab_path = args.bert_path + 'vocab.txt'
    set_seed()

    run_root = Path('../experiments/' + args.run_root)
    DATA_ROOT = Path('../data/')

    if args.mode in ['train', 'validate','validate5','validate55','teacherpred']:
        folds = pd.read_pickle(DATA_ROOT / args.train_file)
        train_fold = folds[folds['fold']!=args.fold]
        valid_fold = folds[folds['fold']==args.fold]
        print(valid_fold.shape)
        print('training fold:', args.fold)
        if args.limit:
            train_fold = train_fold[:args.limit]
            valid_fold = valid_fold[:args.limit]

    if args.mode == 'train':
        if run_root.exists() and args.clean:
            shutil.rmtree(run_root)
        run_root.mkdir(exist_ok=True, parents=True)
        (run_root / 'params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))

        training_set = TrainDataset(train_fold, vocab_path=args.vocab_path, do_lower=True)
        training_loader = DataLoader(training_set, collate_fn=collate_fn,
                                    shuffle=True, batch_size=args.batch_size,
                                    num_workers=args.workers)

        valid_set = TrainDataset(valid_fold, vocab_path=args.vocab_path, mode='test')
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                  num_workers=args.workers)

        model = TweetModel(args.bert_path)
        model.cuda()

        config = BertConfig.from_pretrained(args.bert_path)
        num_layers = config.num_hidden_layers
        optimizer_grouped_parameters = [
            {'params': model.bert.embeddings.parameters(), 'lr': args.lr *
             (args.lr_layerdecay ** num_layers)},
            {'params': model.head.parameters(), 'lr': args.lr},
            {'params': model.bert.pooler.parameters(), 'lr': args.lr}
        ]

        for layer in range(num_layers):
            optimizer_grouped_parameters.append(
                {'params': model.bert.encoder.layer.__getattr__('%d' % (num_layers - 1 - layer)).parameters(),
                 'lr': args.lr * (args.lr_layerdecay ** layer)},
            )
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.lr, eps=args.adam_epsilon)
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O2", verbosity=0)

        total_steps = int(len(train_fold) * args.n_epochs /
                          args.step / args.batch_size)
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
        config = BertConfig(args.bert_path+'config.json')
        model = TweetModel(config=config)
        folds['pred'] = 0
        for fold in range(5):
            valid_fold = folds[folds['fold']==fold]
            load_model(model, run_root / ('best-model-%d.pt' % fold))
            model.cuda()
            valid_set = TrainDataset(valid_fold, vocab_path=args.vocab_path, mode='test', max_len=40)
            valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer),
                                  num_workers=args.workers)
            valid_pred = predict(model, valid_fold, valid_loader, args, progress=True)
            if args.swap:
                valid_set.swap()
                valid_pred2 = predict(
                    model, valid_fold, valid_loader, args, progress=True)
                valid_pred = np.mean([valid_pred, valid_pred2], axis=0)

            folds.loc[valid_fold.index, 'pred'] = valid_pred

        folds.to_csv(run_root / 'new_train_with_pred.csv', index=False)

    elif args.mode == 'teacherpred':
        config = BertConfig(args.bert_path+'config.json')
        model = TweetModel(config=config)
        folds['pred'] = 0
        for fold in range(5):
            valid_fold = folds[folds['fold']==fold]
            load_model(model, run_root / ('best-model-%d.pt' % fold))
            model.cuda()
            valid_set = TrainDataset(valid_fold, vocab_path=args.vocab_path, mode='test', max_len=40)
            valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer),
                                  num_workers=args.workers)
            valid_pred = predict(model, valid_fold, valid_loader, args, progress=True, temperature=args.temperature)
            if args.swap:
                valid_set.swap()
                valid_pred2 = predict(
                    model, valid_fold, valid_loader, args, progress=True, temperature=args.temperature)
                valid_pred = np.mean([valid_pred, valid_pred2], axis=0)

            folds.loc[valid_fold.index, 'pred'] = valid_pred

        folds.to_csv(run_root / args.output_file, index=False)
    
    elif args.mode == 'validate55':
        config = BertConfig(args.bert_path+'config.json')
        model = TweetModel(config=config)
        folds['pred'] = 0
        preds = []
        for fold in range(5):
            load_model(model, run_root / ('best-model-%d.pt' % fold))
            model.cuda()
            valid_set = TrainDataset(valid_fold, vocab_path=args.vocab_path, mode='test', max_len=40)
            valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=partial(collate_fn, tokenizer=tokenizer),
                                      num_workers=args.workers)
            valid_pred = predict(
                model, valid_fold, valid_loader, args, progress=True)
            preds.append(valid_pred)
            if args.swap:
                valid_set.swap()
                valid_pred2 = predict(model, valid_fold, valid_loader, args, progress=True)
                preds.append(valid_pred2)
        preds = np.mean(preds, axis=0)
        m1 = get_metrics(preds, valid_fold['label'], 'all')

    elif args.mode == 'validate':
        valid_set = TrainDataset(valid_fold, vocab_path=args.vocab_path, mode='test')
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                  num_workers=args.workers)
        valid_result = valid_fold.copy()
        config = BertConfig(args.bert_path+'config.json')
        model = TweetModel(config=config)
        load_model(model, run_root / ('best-model-%d.pt' % args.fold))
        model.cuda()
        model = amp.initialize(model, opt_level="O2", verbosity=0)
        if args.multi_gpu == 1:
            model = nn.DataParallel(model)

        all_senti_preds, all_start_preds, all_end_preds = predict(model, valid_fold, valid_loader, args)

        metrics = get_metrics(all_senti_preds, all_start_preds, all_end_preds, valid_fold)

        # valid_result['predict'] = valid_pred
        # valid_result.to_csv(run_root / args.output_file, index=False)

    # elif args.mode == 'predict':
    #     test = pd.read_csv(DATA_ROOT / args.test_file, sep='\t')
    #     test['a'] = test['question1'].apply(lambda x: x.strip())
    #     test['b'] = test['question2'].apply(lambda x: x.strip())

    #     for p, s in replace_operations:
    #         test['a'] = test['a'].apply(lambda x: re.sub(p, s, x))
    #         test['b'] = test['b'].apply(lambda x: re.sub(p, s, x))

    #     test_set = TrainDataset(test, vocab_path=args.vocab_path, swap=False, mode='test',
    #                             max_len=args.max_len)
    #     test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_test_fn,
    #                              num_workers=args.workers)
    #     model = PairModel(args.bert_path)
    #     if args.best_loss:
    #         load_model(model, run_root / ('best-loss-%d.pt' % args.fold), multi2single=False)
    #     else:
    #         load_model(model, run_root / ('best-model-%d.pt' % args.fold), multi2single=False)

    #     model.cuda()
    #     model = amp.initialize(model, None, opt_level="O2", verbosity=0)
    #     if args.multi_gpu == 1:
    #         model = nn.DataParallel(model)
    #     pred1 = predict(model, test, test_loader, args, progress=True)

    #     if args.swap:
    #         test_set.swap()
    #         pred2 = predict(model, test, test_loader, args, progress=True)
    #         np.save(run_root / ('predict-fold%d.npy' % args.fold), np.mean([pred1, pred2], axis=0))
    #         test['pred'] = np.mean([pred1, pred2], axis=0)
    #     else:
    #         test['pred_prob'] = pred1
    #         np.save(run_root / ('predict-fold%d.npy' % args.fold), pred1)
    #     test['pred'] = (test['pred_prob'] > 0.5).astype(int)
    #     run_root = Path('../experiments/' + args.run_root)
    #     test[['qid', 'pred']].to_csv(run_root / ('submission-%d.csv' % args.fold), header=False, sep='\t', index=False)
    #     test.drop(['question1', 'question2'], axis=1).to_csv(run_root / ('analysis-%d.csv' % args.fold), sep='\t', index=False)


def train(args, model: nn.Module, optimizer, scheduler, *,
          train_loader, valid_df, valid_loader, epoch_length, n_epochs=None) -> bool:
    n_epochs = n_epochs or args.n_epochs

    run_root = Path('../experiments/' + args.run_root)
    model_path = run_root / ('model-%d.pt' % args.fold)
    best_model_path = run_root / ('best-model-%d.pt' % args.fold)
    best_model_loss_path = run_root / ('best-loss-%d.pt' % args.fold)
    if best_model_path.exists():
        state, best_valid_score = load_model(model, best_model_path)
        start_epoch = state['epoch']
        best_epoch = start_epoch
    else:
        best_valid_score = 0
        best_valid_loss = 1e10
        start_epoch = 0
        best_epoch = 0
    step = 0
    log = run_root.joinpath('train-%d.log' %
                            args.fold).open('at', encoding='utf8')

    update_progress_steps = int(epoch_length / args.batch_size / 100)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(start_epoch, start_epoch + n_epochs):
        model.train()
        tq = tqdm.tqdm(total=epoch_length)
        losses = []
        mean_loss = 0
        for i, (inputs, targets, starts, ends) in enumerate(train_loader):
            lr = get_learning_rate(optimizer)
            batch_size, length = inputs.size(0), inputs.size(1)
            masks = (inputs > 0).cuda()
            inputs, targets = inputs.cuda(), targets.cuda()
            starts, ends = starts.cuda(), ends.cuda()

            # emb_layer = model.bert.get_input_embeddings()
            # emb = emb_layer(inputs) , input_emb=emb
            senti_out, start_out, end_out = model(inputs, masks)
            # 正常loss
            senti_loss = loss_fn(senti_out, targets) 
            start_loss = loss_fn(start_out, starts)
            end_loss = loss_fn(end_out, ends)
            loss = (0.1*senti_loss+start_loss+end_loss)/ args.step
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            # grad = torch.cat([emb_layer.weight.grad.index_select(0, inputs[i]) for i in range(len(inputs))])
            # grad = grad.view(batch_size,length, -1)
            # # 扰动
            # norm = torch.norm(grad, dim=(1,2))+1e-8
            # noise = args.epsilon * grad / norm.unsqueeze(1).unsqueeze(1)
            # noise[torch.isnan(noise)]=0
            # emb = emb_layer(inputs)
            # adv_outputs = model(None, token_types, masks, input_emb=emb + noise.detach())
            # adv_loss = loss_fn(adv_outputs, targets.view(-1, 1)) / args.step

            # with amp.scale_loss(adv_loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            losses.append(loss.item() * args.step)
            mean_loss = np.mean(losses[-1000:])

            if i>0 and i % update_progress_steps == 0:
                tq.set_description(f'Epoch {epoch}, lr {lr:.6f}')
                tq.update(update_progress_steps*args.batch_size)
                tq.set_postfix(loss=f'{mean_loss:.5f}')
            step += 1
        tq.close()
        valid_metrics = validation(model, valid_df, valid_loader, args)
        write_event(log, step, epoch=epoch, **valid_metrics)
        current_score = valid_metrics['score']
        save_model(model, str(model_path), current_score, epoch + 1)
        if current_score > best_valid_score:
            best_valid_score = current_score
            shutil.copy(str(model_path), str(best_model_path))
            best_epoch = epoch
            print('model saved')
    os.remove(model_path)
    return True


def get_best_pred(start_pred, end_pred, text_len):
    top_start = np.argsort(start_pred)[::-1]
    top_end = np.argsort(end_pred)[::-1]
    preds = []
    for i in range(5):
        for j in range(5):
            if top_start[i]>0 and top_start[i]<text_len+1:  # +1 for cls token
                if top_end[j]>=top_start[i] and top_end[j]<text_len+1:
                    preds.append((top_start[i], top_end[j], start_pred[top_start[i]]+end_pred[top_end[j]]))
    preds = sorted(preds, key=lambda x: x[2], reverse=True)
    return preds[0][0], preds[0][1]

def predict(model: nn.Module, valid_df, valid_loader, args, progress=False) -> Dict[str, float]:
    # run_root = Path('../experiments/' + args.run_root)
    model.eval()
    all_end_pred, all_senti_pred, all_start_pred = [], [], []
    if progress:
        tq = tqdm.tqdm(total=len(valid_df))
    with torch.no_grad():
        for inputs, _, _, _ in valid_loader:
            if progress:
                batch_size = inputs.size(0)
                tq.update(batch_size)
            masks = (inputs > 0).cuda()
            inputs = inputs.cuda()
            senti_out, start_out, end_out = model(inputs, masks)

            senti_pred = torch.argmax(senti_out, dim=-1)
            # start_out = torch.argmax(start_out[:,1:], dim=-1)
            # end_out = torch.argmax(end_out[:,1:], dim=-1)
            # predictions = outputs.clamp(min=0, max=2)
            all_senti_pred.append(senti_pred.cpu().numpy())
            start_out = torch.softmax(start_out, dim=-1).cpu().numpy()
            end_out = torch.softmax(end_out, dim=-1).cpu().numpy()
            masks = masks.cpu().numpy().astype(np.int)
            for idx in range(len(senti_pred)):
                start, end = get_best_pred(start_out[idx,:], end_out[idx,:], np.sum(masks[idx,:])-2)
                all_start_pred.append(start-1)
                all_end_pred.append(end-1)

    all_senti_pred = np.concatenate(all_senti_pred)
    if progress:
        tq.close()
    return all_senti_pred, all_start_pred, all_end_pred


def validation(model: nn.Module, valid_df, valid_loader, args, save_result=False, progress=False):
    run_root = Path('../experiments/' + args.run_root)

    all_senti_preds, all_start_preds, all_end_preds = predict(model, valid_df, valid_loader, args)
    

    # if save_result:
    #     np.save(run_root / 'prediction_fold{}.npy'.format(args.fold),
    #             all_senti_preds)
    #     np.save(run_root / 'target_fold{}.npy'.format(args.fold), all_targets)

    metrics = get_metrics(all_senti_preds, all_start_preds, all_end_preds, valid_df)

    
    return metrics


def get_metrics(all_senti_preds, all_start_preds, all_end_preds, valid_df):
    all_senti_labels = valid_df['senti_label'].values
    metrics = dict()
    metrics['loss'] = 0    
    metrics['senti_acc'] = accuracy_score(all_senti_labels, all_senti_preds)
    
    invert_maps = valid_df['invert_map'].tolist()
    selected_texts = valid_df['selected_text'].tolist()
    texts = valid_df['text'].tolist()
    
    score = 0
    for idx in range(len(texts)):
        text = texts[idx]
        words = text.lower().split()
        invert_map = invert_maps[idx]

        start_word = invert_map[all_start_preds[idx]]
        end_word = invert_map[all_end_preds[idx]]
        selected_text_pred = words[start_word:end_word+1]
        selected_text_label = selected_texts[idx].lower().split()
        score+=jaccard_list(selected_text_pred, selected_text_label)
        if idx<5:
            print(' '.join(selected_text_pred), ' '.join(selected_text_label))
    metrics['score'] = score/len(texts)
    print('senti_acc:', metrics['senti_acc'], 'jaccard:', metrics['score'])
    print(confusion_matrix(all_senti_labels, all_senti_preds))
    return metrics


if __name__ == '__main__':
    main()
