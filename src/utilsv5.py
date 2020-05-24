# BucketSampler from https://github.com/PetrochukM/PyTorch-NLP

from torch.utils.data.sampler import Sampler
import heapq
import math
import numpy as np
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import RandomSampler
import random
import torch
import collections
import inspect
from torch import nn
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime
import json
import torch.nn.functional as F
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             log_loss, roc_auc_score)
from transformers import BasicTokenizer
import re

pattern = r"\w+[.]{1,5}\w+"
pattern = re.compile(pattern)

def prepare(text):
    words = text.split()
    retval, first_char, invert_map = [], [], []
    current_pos = 0
    for w in words:
        word_ret = [""]
        word_invert = [current_pos]
        for p, c in enumerate(w):
            if c in ['.',',','!','?','(',')',';',':','-','=',"/","<","`"]:
                if word_ret[-1]=="":
                    word_ret[-1]+=c
                    word_invert[-1]=current_pos+p
                else:
                    word_ret.append(c)
                    word_invert.append(current_pos+p)
                word_ret.append("")
                word_invert.append(current_pos+p+1)
            else:
                word_ret[-1]+=c
        if len(word_ret[-1])==0:
            word_ret.pop(-1)
            word_invert.pop(-1)
        word_first = [False if i>0 else True for i in range(len(word_ret)) ]
        retval.extend(word_ret)
        first_char.extend(word_first)
        invert_map.extend(word_invert)
        current_pos+=len(w)+1
    assert len(retval)==len(first_char)
    return retval, first_char, invert_map

def decode(tokens, first_char, start, end):
    retval = ""
    for i in range(start, end+1):
        if first_char[i]:
            retval+= (" "+tokens[i])
        else:
            retval += tokens[i]
    return " ".join(retval.split())


def map_to_word(preds, df, args, softmax=True):
    invert_maps = df['invert_map'].tolist()
    words = df['words'].tolist()
    retval = []
    for idx in range(len(invert_maps)):
        word = words[idx]
        temp = torch.zeros(len(word))
        if softmax:
            pred = torch.softmax(preds[idx], dim=-1)
            for p in range(len(invert_maps[idx])):
                temp[invert_maps[idx][p]]+= pred[p+args.offset]
        else:
            pred = preds[idx]
            for p in range(len(invert_maps[idx])):
                # temp[invert_maps[idx][p]]+= pred[p+args.offset]
                temp[invert_maps[idx][p]] = max(temp[invert_maps[idx][p]], pred[p+args.offset])
        retval.append(temp)
    return retval


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text

def ensemble_words(word_preds):
    final_word_preds = []
    model_num = len(word_preds)
    for idx in range(len(word_preds[0])):
        temp = []
        for m_idx in range(model_num):
            temp+=list(set(word_preds[m_idx][idx].split()))
        word_count = collections.Counter(temp)
        temp = [w for w in word_count.keys() if word_count[w]>=model_num*0.4]
        final_word_preds.append(' '.join(temp))
    return final_word_preds

def ensemble(whole_preds, start_preds, end_preds, inst_preds, df, softmax=False):
    # 在logit层面融合
    all_whole_preds, all_end_pred, all_start_pred, all_inst_preds = [], [], [], []
    model_num = len(start_preds)
    for b_idx in range(len(start_preds[0])):
        # merge one batch
        whole_out = whole_preds[0][b_idx]
        if softmax:
            start_out = torch.softmax(start_preds[0][b_idx], axis=-1)
            end_out = torch.softmax(end_preds[0][b_idx], axis=-1)
        else:
            start_out = start_preds[0][b_idx]
            end_out = end_preds[0][b_idx]
        inst_out = inst_preds[0][b_idx]
        for m_idx in range(1, model_num):
            whole_out += whole_preds[m_idx][b_idx]
            if softmax:
                start_out += torch.softmax(start_preds[m_idx][b_idx], axis=-1)
                end_out += torch.softmax(end_preds[m_idx][b_idx], axis=-1)
            else:
                start_out += start_preds[m_idx][b_idx]
                end_out += end_preds[m_idx][b_idx]
            inst_out += inst_preds[m_idx][b_idx]
        
        whole_out = whole_out/model_num
        start_out = start_out/model_num
        end_out = end_out/model_num
        inst_out = inst_out/model_num

        all_whole_preds.append(whole_out)
        all_start_pred.append(start_out)
        all_end_pred.append(end_out)
        all_inst_preds.append(inst_out)
    return all_whole_preds, all_start_pred, all_end_pred, all_inst_preds


def get_best_pred(start_pred, end_pred):
    top_start = np.argsort(start_pred.numpy())[::-1]
    top_end = np.argsort(end_pred.numpy())[::-1]

    preds = []
    for start in top_start[:30]:
        for end in top_end[:30]:
            if end < start:
                continue
            preds.append((start, end, start_pred[start]+end_pred[end]))
    preds = sorted(preds, key=lambda x: x[2], reverse=True)
    if len(preds)==0:
        print(top_start[:30], top_end[:30])
        return 0, 0
    else:
        # scores, spans = 0, []
        # for i in range(len(preds)):
        #     spans.append((preds[i][0], preds[i][1]))
        #     scores+=preds[i][2].item()
        #     if scores>1:
        #         break
        # return spans, scores
        return preds[0][0], preds[0][1], preds[0][2].item()

def get_predicts_from_word_logits(all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, valid_df, args):
    all_senti_labels = valid_df['senti_label'].values
    all_words = valid_df['words'].tolist()
    first_chars = valid_df['first_char'].tolist()

    word_preds, inst_word_preds, scores = [], [], []
    for idx in range(len(all_words)):
        words = all_words[idx]
        start_word, end_word, score = get_best_pred(all_start_preds[idx], all_end_preds[idx])
                
        # spans, score = get_best_pred(all_start_preds[idx], all_end_preds[idx])


        inst_pred = all_inst_preds[idx]
        inst_word_pred = ' '.join([words[p] for p in range(len(words)) if inst_pred[p]>0.95])
        word_pred = decode(words, first_chars[idx], start_word, end_word)
        # word_pred = ''
        # for (s, e) in spans:
        #     word_pred += ' '+decode(words, first_chars[idx], s, e)

        if all_whole_preds[idx]>0.5:
            word_pred = decode(words, first_chars[idx], 0, len(words)-1)
            inst_word_pred = word_pred
            
        if args.post:
            if all_senti_labels[idx]==1:
                word_pred = decode(words, first_chars[idx], 0, len(words)-1)
                inst_word_pred = word_pred

        word_preds.append(word_pred)
        inst_word_preds.append(inst_word_pred)
        scores.append(score)
    return word_preds, inst_word_preds, scores


def get_predicts_from_token_logits(all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, valid_df, args, softmax=False):
    all_start_preds = map_to_word(all_start_preds, valid_df, args, softmax=softmax)
    all_end_preds = map_to_word(all_end_preds, valid_df, args, softmax=softmax)
    all_inst_preds = map_to_word(all_inst_preds, valid_df, args, softmax=False)
    word_preds, inst_word_preds, scores = get_predicts_from_word_logits(all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, valid_df, args)
    return word_preds, inst_word_preds, scores


def get_loss(pred, label):
    loss_fn = torch.nn.CrossEntropyLoss()
    retval = 0
    for idx in range(len(label)):
        retval+= loss_fn(pred[idx].unsqueeze(0), torch.LongTensor([label[idx]])).item()
    return retval/len(label)


def evaluate(word_preds, valid_df, args=None):  #all_senti_preds, all_start_preds, all_end_preds, 
    metrics = dict()
    metrics['loss'] = 0
    invert_maps = valid_df['invert_map'].tolist()
    # starts = valid_df['start'].tolist()
    # ends = valid_df['end'].tolist()
    texts = valid_df['text'].tolist()
    all_senti_labels = valid_df['senti_label'].values
    selected_texts = valid_df['selected_text'].tolist()

    clean_score_word, dirty_score_word = 0, 0

    for idx in range(len(texts)):
        text = texts[idx]
        words = text.lower().split()
        invert_map = invert_maps[idx]
        
        # start_word_label, end_word_label = invert_map[starts[idx]], invert_map[ends[idx]]
        # clean_label = ' '.join(words[start_word_label: end_word_label+1])
        
        # clean_score_word += jaccard_string(word_preds[idx], clean_label)
        # clean_score_token += jaccard_string(token_preds[idx], clean_label)

        # dirty label & dirty score
        dirty_label = selected_texts[idx]
        dirty_score_word += jaccard_string(word_preds[idx], dirty_label)
        # dirty_score_token += jaccard_string(token_preds[idx], dirty_label)

        if idx < 10:
            print(word_preds[idx], " || ", dirty_label)

    # metrics['clean_score_word'] = clean_score_word/len(texts)
    metrics['dirty_score_word'] = dirty_score_word/len(texts)

    print(
        #   'clean word:', metrics['clean_score_word'], 
          'dirty word:', metrics['dirty_score_word'])
    return metrics


def jaccard_string(s1, s2):
    a = set(s1.lower().split())
    b = set(s2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def jaccard_list(l1, l2):
    a = set(l1)
    b = set(l2)
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def binary_focal_loss(pred,
                      target,
                      weight=None,
                      gamma=2.0,
                      alpha=0.25,
                      reduction='mean',
                      avg_factor=None):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    loss = torch.mean(loss)
    return loss


def pad_sequence(sequences, batch_first=False, padding_value=0, pad_to_left=False):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            if pad_to_left:
                out_tensor[i, -length:, ...] = tensor
            else:
                out_tensor[i, :length, ...] = tensor
        else:
            if pad_to_left:
                out_tensor[-length:, i, ...] = tensor
            else:
                out_tensor[:length, i, ...] = tensor
    return out_tensor


def set_seed(seed=6750):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model: nn.Module, path: Path) -> Tuple:
    state = torch.load(str(path))
    model.load_state_dict(state['model'])
    print('Loaded model from epoch {epoch}'.format(**state))
    return state, state['best_valid_loss']


def save_model(model, model_path, current_score, ep):
    return torch.save({
        'model': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'epoch': ep,
        'best_valid_loss': current_score
    }, str(model_path))


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]

    lr = lr[0]

    return lr


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return True


def _biggest_batches_first(o):
    return sum([t.numel() for t in get_tensors(o)])


def _identity(e):
    return e

def get_tensors(object_):
    """ Get all tensors associated with ``object_``
    Args:
        object_ (any): Any object to look for tensors.
    Returns:
        (list of torch.tensor): List of tensors that are associated with ``object_``.
    """
    if torch.is_tensor(object_):
        return [object_]
    elif isinstance(object_, (str, float, int)):
        return []

    tensors = set()

    if isinstance(object_, collections.abc.Mapping):
        for value in object_.values():
            tensors.update(get_tensors(value))
    elif isinstance(object_, collections.abc.Iterable):
        for value in object_:
            tensors.update(get_tensors(value))
    else:
        members = [
            value for key, value in inspect.getmembers(object_)
            if not isinstance(value, (collections.abc.Callable, type(None)))
        ]
        tensors.update(get_tensors(members))

    return tensors


class ShuffleBatchSampler(BatchSampler):
    """Wraps another sampler to yield a mini-batch of indices.
    The ``ShuffleBatchSampler`` adds ``shuffle`` on top of
    ``torch.utils.data.sampler.BatchSampler``.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if its size would be
            less than ``batch_size``.
        shuffle (bool, optional): If ``True``, the sampler will shuffle the batches.
    Example:
        >>> import random
        >>> from torchnlp.samplers import SortedSampler
        >>>
        >>> random.seed(123)
        >>>
        >>> list(ShuffleBatchSampler(SortedSampler(range(10)), batch_size=3, drop_last=False))
        [[6, 7, 8], [9], [3, 4, 5], [0, 1, 2]]
        >>> list(ShuffleBatchSampler(SortedSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [6, 7, 8], [3, 4, 5]]
    """

    def __init__(
            self,
            sampler,
            batch_size,
            drop_last,
            shuffle=True,
    ):
        self.shuffle = shuffle
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        # NOTE: This is not data
        batches = list(super().__iter__())
        if self.shuffle:
            random.shuffle(batches)

        return iter(batches)


class SortedSampler(Sampler):
    """Samples elements sequentially, always in the same order.
    Args:
        data (iterable): Iterable data.
        sort_key (callable): Specifies a function of one argument that is used to extract a
            numerical comparison key from each list element.
    Example:
        >>> list(SortedSampler(range(10), sort_key=lambda i: -i))
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    """

    def __init__(self, data, sort_key=_identity):
        super().__init__(data)
        self.data = data
        self.sort_key = sort_key
        zip = [(i, self.sort_key(row)) for i, row in enumerate(self.data)]
        zip = sorted(zip, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)


class BucketBatchSampler(object):
    """ Batches are sampled from sorted buckets of data.

    We use a bucketing technique from ``torchtext``. First, partition data in buckets of size
    100 * ``batch_size``. The examples inside the buckets are sorted using ``sort_key`` and batched.

    **Background**

        BucketBatchSampler is similar to a BucketIterator found in popular libraries like `AllenNLP`
        and `torchtext`. A BucketIterator pools together examples with a similar size length to
        reduce the padding required for each batch. BucketIterator also includes the ability to add
        noise to the pooling.

        **AllenNLP Implementation:**
        https://github.com/allenai/allennlp/blob/e125a490b71b21e914af01e70e9b00b165d64dcd/allennlp/data/iterators/bucket_iterator.py

        **torchtext Implementation:**
        https://github.com/pytorch/text/blob/master/torchtext/data/iterator.py#L225

    Args:
        data (iterable): Data to sample from.
        batch_size (int): Size of mini-batch.
        sort_key (callable): specifies a function of one argument that is used to extract a
          comparison key from each list element
        drop_last (bool): If ``True``, the sampler will drop the last batch if its size would be
            less than ``batch_size``.
        biggest_batch_first (callable or None, optional): If a callable is provided, the sampler
            approximates the memory footprint of tensors in each batch, returning the 5 biggest
            batches first. Callable must return a number, given an example.

            This is largely for testing, to see how large of a batch you can safely use with your
            GPU. This will let you try out the biggest batch that you have in the data `first`, so
            that if you're going to run out of memory, you know it early, instead of waiting
            through the whole epoch to find out at the end that you're going to crash.

            Credits:
            https://github.com/allenai/allennlp/blob/3d100d31cc8d87efcf95c0b8d162bfce55c64926/allennlp/data/iterators/bucket_iterator.py#L43
        bucket_size_multiplier (int): Batch size multiplier to determine the bucket size.
        shuffle (bool, optional): If ``True``, the sampler will shuffle the batches.

    Example:
        >>> list(BucketBatchSampler(range(10), batch_size=3, drop_last=False))
        [[9], [3, 4, 5], [6, 7, 8], [0, 1, 2]]
        >>> list(BucketBatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    """

    def __init__(
            self,
            data,
            batch_size,
            drop_last,
            sort_key=_identity,
            biggest_batches_first=_biggest_batches_first,
            bucket_size_multiplier=100,
            shuffle=True,
    ):
        self.biggest_batches_first = biggest_batches_first
        self.sort_key = sort_key
        self.bucket_size_multiplier = bucket_size_multiplier
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.data = data
        self.shuffle = shuffle

        self.bucket_size_multiplier = bucket_size_multiplier
        self.bucket_sampler = BatchSampler(
            RandomSampler(data), batch_size * bucket_size_multiplier, False)

    def __iter__(self):

        def get_batches():
            """ Get bucketed batches """
            for bucket in self.bucket_sampler:
                for batch in ShuffleBatchSampler(
                        SortedSampler(
                            bucket, lambda i: self.sort_key(self.data[i])),
                        self.batch_size,
                        drop_last=self.drop_last,
                        shuffle=self.shuffle):
                    batch = [bucket[i] for i in batch]

                    # Should only be triggered once
                    if len(batch) < self.batch_size and self.drop_last:
                        continue

                    yield batch

        if self.biggest_batches_first is None:
            return get_batches()
        else:
            batches = list(get_batches())
            biggest_batches = heapq.nlargest(
                5,
                range(len(batches)),
                key=lambda i: sum([self.biggest_batches_first(self.data[j]) for j in batches[i]]))
            front = [batches[i] for i in biggest_batches]
            # Remove ``biggest_batches`` from data
            for i in sorted(biggest_batches, reverse=True):
                batches.pop(i)
            # Move them to the front
            batches[0:0] = front
            return iter(batches)

    def __len__(self):
        if self.drop_last:
            return len(self.data) // self.batch_size
        else:
            return math.ceil(len(self.data) / self.batch_size)


def write_event(log, step: int, epoch=None, **data):
    data['step'] = step
    data['dt'] = datetime.now().isoformat()
    data['epoch'] = epoch
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()
