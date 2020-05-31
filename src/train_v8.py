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
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import (AdamW, get_cosine_schedule_with_warmup,
                                       get_linear_schedule_with_warmup,
                                       get_cosine_with_hard_restarts_schedule_with_warmup)

from utilsv5 import (binary_focal_loss, get_learning_rate, jaccard_list, get_best_pred, ensemble, ensemble_words, prepare,
                   load_model, save_model, set_seed, write_event, evaluate, get_predicts_from_token_logits, map_to_word)
from dataset4 import TrainDataset, MyCollator

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
            self.backup = {}


class TweetModel(nn.Module):

    def __init__(self, pretrain_path=None, dropout=0.2, config=None):
        super(TweetModel, self).__init__()
        if config is not None:
            self.bert = AutoModel.from_config(config)
        else:
            config = AutoConfig.from_pretrained(pretrain_path, output_hidden_states=True)
            self.bert = AutoModel.from_pretrained(
                pretrain_path, cache_dir=None, config=config)
        
        self.cnn =  nn.Conv1d(self.bert.config.hidden_size, self.bert.config.hidden_size, 3, padding=1)

        # self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size//2, num_layers=2,
        #                     batch_first=True, bidirectional=True)
        self.gelu = nn.GELU()

        self.whole_head = nn.Linear(self.bert.config.hidden_size, 2)
        self.se_head = nn.Linear(self.bert.config.hidden_size, 2)
        self.inst_head = nn.Linear(self.bert.config.hidden_size, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, inputs, masks, token_type_ids=None, input_emb=None):
        _, pooled_output, hs = self.bert(
            inputs, masks, token_type_ids=token_type_ids, inputs_embeds=input_emb)

        seq_output = hs[-1] #+hs[-3]

        # senti = seq_output[:,1,:]
        # seq_output = seq_output*senti.unsqueeze(1)

        whole_out = self.whole_head(self.dropout(F.adaptive_avg_pool1d(seq_output.permute(0,2,1), 1).squeeze(-1)))

        seq_output = self.gelu(self.cnn(seq_output.permute(0,2,1)).permute(0,2,1))
        
        se_out = self.se_head(self.dropout(seq_output))  #()
        inst_out = self.inst_head(self.dropout(seq_output))
        return whole_out, se_out[:, :, 0], se_out[:, :, 1], inst_out


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('mode', choices=['train', 'validate', 'predict', 'predict5',
                         'validate5', 'validate52'])
    arg('run_root')
    arg('--batch-size', type=int, default=32)
    arg('--step', type=int, default=1)
    arg('--workers', type=int, default=2)
    arg('--lr', type=float, default=0.00002)
    arg('--clean', action='store_true')
    arg('--n-epochs', type=int, default=3)
    arg('--limit', type=int)
    arg('--fold', type=int, default=0)
    arg('--multi-gpu', type=int, default=0)

    arg('--bert-path', type=str, default='../../bert_models/roberta_base/')
    arg('--train-file', type=str, default='train_folds.csv')
    arg('--local-test', type=str, default='localtest_roberta.pkl')
    arg('--test-file', type=str, default='test.csv')
    arg('--output-file', type=str, default='result.csv')
    arg('--no-neutral', action='store_true')
    arg('--holdout', action='store_true')
    arg('--distill', action='store_true')

    arg('--epsilon', type=float, default=0.3)
    arg('--max-len', type=int, default=200)
    arg('--fp16', action='store_true')
    arg('--lr-layerdecay', type=float, default=1.0)
    arg('--max_grad_norm', type=float, default=-1.0)
    arg('--weight_decay', type=float, default=0.0)
    arg('--adam-epsilon', type=float, default=1e-6)
    arg('--offset', type=int, default=5)
    arg('--best-loss', action='store_true')
    arg('--abandon', action='store_true')
    arg('--post', action='store_true')
    arg('--smooth', action='store_true')
    arg('--temperature', type=float, default=1)

    args = parser.parse_args()
    args.vocab_path = args.bert_path
    set_seed(42)

    run_root = Path('../experiments/' + args.run_root)
    DATA_ROOT = Path('../input/tweet-sentiment-extraction/')
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_path, cache_dir=None, do_lower_case=False)
    args.tokenizer = tokenizer
    if args.bert_path.find('roberta'):
        collator = MyCollator()
    else:
        # this is for bert models
        collator = MyCollator(token_pad_value=0, type_pad_value=1)
    folds = pd.read_csv(DATA_ROOT / args.train_file)
    if args.mode in ['train', 'validate', 'validate5', 'validate55', 'teacherpred']:
        # folds = pd.read_pickle(DATA_ROOT / args.train_file)
        train_fold = folds[folds['kfold'] != args.fold]
        if args.abandon:
            train_fold = train_fold[train_fold['label_jaccard']>0.6]
        if args.no_neutral:
            train_fold = train_fold[train_fold['sentiment']!='neutral']
        valid_fold = folds[folds['kfold'] == args.fold]
        # remove pseudo samples
        if 'type' in valid_fold.columns.tolist():
            valid_fold = valid_fold[valid_fold['type']=='normal']
        print(valid_fold.shape)
        print('training fold:', args.fold)
        if args.limit:
            train_fold = train_fold[:args.limit]
            valid_fold = valid_fold[:args.limit]

    old_data = pd.read_csv(DATA_ROOT/'tweet_dataset.csv')
    old_data.dropna(subset=['text'],inplace=True)
    old_data['text'] = old_data['text'].apply(lambda x: ' '.join(x.strip().split()))
    old_data.drop_duplicates(subset=['text'], inplace=True)
    old_data.rename(index=str, columns={'sentiment':'sentiment2'}, inplace=True)

    if args.mode == 'train':
        if run_root.exists() and args.clean:
            shutil.rmtree(run_root)
        run_root.mkdir(exist_ok=True, parents=True)

        training_set = TrainDataset(train_fold, old_data, tokenizer=tokenizer, smooth=args.smooth, offset=args.offset)
        training_loader = DataLoader(training_set, collate_fn=collator,
                                     shuffle=True, batch_size=args.batch_size,
                                     num_workers=args.workers)
        
        valid_set = TrainDataset(valid_fold, old_data, tokenizer=tokenizer, mode='test', offset=args.offset)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
                                  num_workers=args.workers)

        model = TweetModel(args.bert_path)
        model.cuda()

        config = RobertaConfig.from_pretrained(args.bert_path)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0},
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
        config = AutoConfig.from_pretrained(args.bert_path, output_hidden_states=True)
        model = TweetModel(config=config)
        for fold in range(5):
            valid_fold = folds[folds['kfold']==fold]
            load_model(model, run_root / ('best-model-%d.pt' % fold))
            model.cuda()

            valid_set = TrainDataset(valid_fold, old_data, tokenizer=tokenizer, mode='test', offset=args.offset)
            valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
                                      num_workers=args.workers)
            fold_whole_preds, fold_start_pred, fold_end_pred, fold_inst_preds = predict(
                model, valid_fold, valid_loader, args, progress=True)
            word_preds, _, scores = get_predicts_from_token_logits(fold_whole_preds, fold_start_pred, fold_end_pred, fold_inst_preds, valid_fold, args, softmax=True)
            metrics = evaluate(word_preds, fold_whole_preds, valid_fold, args)

            dis_start_pred, dis_end_pred = [], []
            
            for i in range(len(fold_start_pred)):
                dis_start_pred.append(torch.softmax(fold_start_pred[i][args.offset:]/args.temperature, dim=-1).numpy())
                dis_end_pred.append(torch.softmax(fold_end_pred[i][args.offset:]/args.temperature, dim=-1).numpy())
            assert len(valid_fold)==len(dis_start_pred)
            folds.loc[valid_fold.index, 'start_pred'] = dis_start_pred
            folds.loc[valid_fold.index, 'end_pred'] = dis_end_pred
            folds.loc[valid_fold.index, 'pred'] = word_preds
            folds.loc[valid_fold.index, 'score'] = scores
            folds.loc[valid_fold.index, 'whole_pred'] = fold_whole_preds
        folds.to_pickle(DATA_ROOT/'preds.pkl')
    
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
        if args.holdout:
            valid_fold = pd.read_pickle(DATA_ROOT / args.local_test)
        valid_set = TrainDataset(valid_fold, old_data, tokenizer=tokenizer, mode='test', offset=args.offset)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
                                  num_workers=args.workers)
        valid_result = valid_fold.copy()
        config = AutoConfig.from_pretrained(args.bert_path)
        model = TweetModel(config=config)
        load_model(model, run_root / ('best-model-%d.pt' % args.fold))
        model.cuda()
        if args.multi_gpu == 1:
            model = nn.DataParallel(model)

        all_whole_preds, all_start_preds, all_end_preds, all_inst_preds = predict(
            model, valid_fold, valid_loader, args, progress=True)
        word_preds, inst_word_preds, scores = get_predicts_from_token_logits(all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, valid_fold, args, softmax=True)
        metrics = evaluate(word_preds, valid_fold, args)
        # metrics = evaluate(inst_word_preds, valid_fold, args)
        # valid_fold['pred'] = word_preds
        # valid_fold['score'] = scores
        # valid_fold['inst_pred'] = inst_word_preds
        # valid_fold.to_csv(run_root/('pred-%d.csv'%args.fold), sep='\t', index=False)

    elif args.mode in ['predict', 'predict5']:
        test = pd.read_csv(DATA_ROOT /args.test_file)       

        test_set = TrainDataset(test, tokenizer=tokenizer, mode='test')
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
                                 num_workers=args.workers)
        config = AutoConfig.from_pretrained(args.bert_path)
        model = TweetModel(config=config)
        
        if args.mode == 'predict':
            load_model(model, run_root / ('best-model-%d.pt' % args.fold))
            class Args:
                post = True
                tokenizer = RobertaTokenizer.from_pretrained(args.bert_path)
                offset = 4
                batch_size = 16
                workers = 1
                bert_path = args.bert_path
            args = Args()
            model.cuda()

            all_whole_preds, all_start_preds, all_end_preds, all_inst_preds = predict(
                model, test, test_loader, args, progress=True)
            word_preds, inst_word_preds, scores = get_predicts_from_token_logits(all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, test, args)
            metrics = evaluate(word_preds, test, args)
        
        if args.mode == 'predict5':
            all_whole_preds, all_start_preds, all_end_preds, all_inst_preds = [], [], [], []
            for fold in range(5):
                load_model(model, run_root / ('best-model-%d.pt' % fold))
                model.cuda()
                fold_whole_preds, fold_start_preds, fold_end_preds, fold_inst_preds = predict(model, test, test_loader, args, progress=True, for_ensemble=True)
                # fold_start_preds = map_to_word(fold_start_preds, test, args, softmax=False)
                # fold_end_preds = map_to_word(fold_end_preds, test, args, softmax=False)
                # fold_inst_preds = map_to_word(fold_inst_preds, test, args, softmax=False)

                all_whole_preds.append(fold_whole_preds)
                all_start_preds.append(fold_start_preds)
                all_end_preds.append(fold_end_preds)
                all_inst_preds.append(fold_inst_preds)

            all_whole_preds, all_start_preds, all_end_preds, all_inst_preds = ensemble(all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, test, softmax=True)
            word_preds, inst_word_preds, scores = get_predicts_from_token_logits(all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, test, args)

        test['selected_text'] = word_preds
        test['score'] = scores
        test[['textID','selected_text','score']].to_csv('submission.csv', index=False)

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
    kl_fn = nn.KLDivLoss(reduction='batchmean')
    ce_fn = nn.CrossEntropyLoss(ignore_index=-100)
    bce_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.NLLLoss()
    fgm = FGM(model)
    for epoch in range(start_epoch, start_epoch + n_epochs):
        model.train()
        tq = tqdm.tqdm(total=epoch_length)
        losses = []
        mean_loss = 0
        for i, (tokens, types, masks, targets, starts, ends, hard_starts, hard_ends, inst, all_sentence) in enumerate(train_loader):
            lr = get_learning_rate(optimizer)
            batch_size, length = tokens.size(0), tokens.size(1)
            masks = masks.cuda()
            tokens, targets = tokens.cuda(), targets.cuda()
            types = types.cuda()
            starts, ends, inst = starts.cuda(), ends.cuda(), inst.cuda()
            hard_starts, hard_ends = hard_starts.cuda(), hard_ends.cuda()
            all_sentence = all_sentence.cuda()

            whole_out, start_out, end_out, inst_out = model(tokens, masks, types)
            # start_out = start_out.masked_fill(~masks.bool(), -10000.0)
            # end_out = end_out.masked_fill(~masks.bool(), -10000.0)
            # 正常loss
            whole_loss = ce_fn(whole_out, all_sentence)
            start_out = torch.log_softmax(start_out, dim=-1)
            end_out = torch.log_softmax(end_out, dim=-1)
            start_loss = ce_fn(start_out, hard_starts)
            end_loss = ce_fn(end_out, hard_ends)
            if args.distill:
                # soft label loss
                start_loss += kl_fn(start_out, starts)
                end_loss += kl_fn(end_out, ends)

            inst_loss = ce_fn(inst_out.permute(0,2,1), inst)
            loss = (start_loss+end_loss)+inst_loss+whole_loss

            loss /= args.step

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            fgm.attack() 
            whole_out, start_out, end_out, inst_out = model(tokens, masks, types)
            # start_out = start_out.masked_fill(~masks.bool(), -10000.0)
            # end_out = end_out.masked_fill(~masks.bool(), -10000.0)
            whole_loss = ce_fn(whole_out, all_sentence)
            
            start_out = torch.log_softmax(start_out, dim=-1)
            end_out = torch.log_softmax(end_out, dim=-1)
            start_loss = ce_fn(start_out, hard_starts)
            end_loss = ce_fn(end_out, hard_ends)
            if args.distill:
                # soft label loss
                start_loss += kl_fn(start_out, starts)
                end_loss += kl_fn(end_out, ends)

            inst_loss = ce_fn(inst_out.permute(0,2,1), inst)
            loss = (start_loss+end_loss)+inst_loss+whole_loss

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                
            fgm.restore()
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


def predict(model: nn.Module, valid_df, valid_loader, args, progress=False) -> Dict[str, float]:
    # run_root = Path('../experiments/' + args.run_root)
    model.eval()
    all_end_pred, all_whole_pred, all_start_pred, all_inst_out = [], [], [], []
    if progress:
        tq = tqdm.tqdm(total=len(valid_df))
    with torch.no_grad():
        for tokens, types, masks, _, _, _, _, _, _, _ in valid_loader:
            if progress:
                batch_size = tokens.size(0)
                tq.update(batch_size)
            masks = masks.cuda()
            tokens = tokens.cuda()
            types = types.cuda()
            whole_out, start_out, end_out, inst_out = model(tokens, masks, types)
            start_out = start_out.masked_fill(~masks.bool(), -1000)
            end_out= end_out.masked_fill(~masks.bool(), -1000)           
            all_whole_pred.append(torch.softmax(whole_out, dim=-1)[:,1].cpu().numpy())
            inst_out = torch.softmax(inst_out, dim=-1)
            for idx in range(len(start_out)):
                all_start_pred.append(start_out[idx,:].cpu())
                all_end_pred.append(end_out[idx,:].cpu())
                all_inst_out.append(inst_out[idx,:,1].cpu())
            assert all_start_pred[-1].dim()==1

    all_whole_pred = np.concatenate(all_whole_pred)
    
    if progress:
        tq.close()
    return all_whole_pred, all_start_pred, all_end_pred, all_inst_out


def validation(model: nn.Module, valid_df, valid_loader, args, save_result=False, progress=False):
    run_root = Path('../experiments/' + args.run_root)

    all_whole_preds, all_start_preds, all_end_preds, all_inst_out = predict(
        model, valid_df, valid_loader, args)
    word_preds, inst_preds, scores = get_predicts_from_token_logits(all_whole_preds, all_start_preds, all_end_preds, all_inst_out, valid_df, args)
    # metrics = evaluate(inst_preds, valid_df, args)
    metrics = evaluate(word_preds, all_whole_preds, valid_df, args)
    return metrics


if __name__ == '__main__':
    main()
