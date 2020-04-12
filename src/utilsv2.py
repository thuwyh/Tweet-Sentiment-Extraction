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


def ensemble(senti_preds, start_preds, end_preds, df):
    # merge senti
    # todo

    all_end_pred, all_start_pred = [], []
    tokens = df['tokens'].tolist()
    model_num = len(start_preds)
    for b_idx in range(len(start_preds[0])):
        # merge one batch
        start_out = start_preds[0][b_idx]
        end_out = end_preds[0][b_idx]
        for m_idx in range(1, model_num):
            start_out += start_preds[m_idx][b_idx]
            end_out += end_preds[m_idx][b_idx]
        start_out = start_out/model_num
        end_out = end_out/model_num

        start_out = torch.softmax(start_out, dim=-1).numpy()
        end_out = torch.softmax(end_out, dim=-1).numpy()

        for idx in range(len(end_out)):
            start, end = get_best_pred(
                start_out[idx, :], end_out[idx, :], len(tokens[len(all_end_pred)]), offset=4)
            all_start_pred.append(start)
            all_end_pred.append(end)
    return all_start_pred, all_end_pred


def get_best_pred(start_pred, end_pred, text_len, offset=3):
    top_start = np.argsort(start_pred)[::-1] # start can not be the last token  [offset:text_len+offset]
    top_end = np.argsort(end_pred)[::-1]  # [offset:text_len+offset]
    preds = []
    for start in top_start[:30]:
        for end in top_end[:30]:
            if start<offset:
                continue
            if end<offset:
                continue
            if end >= start:
                preds.append((start, end, start_pred[start]+end_pred[end]))
    preds = sorted(preds, key=lambda x: x[2], reverse=True)
    # top_start = np.argmax(start_pred)-offset
    # top_end = np.argmax(end_pred)-offset
    if len(preds)==0:
        print(top_start[:30], top_end[:30])
        return 0, 0
    else:
        return preds[0][0]-offset, preds[0][1]-offset
    # return top_start, top_end


def get_predicts(all_senti_preds, all_start_preds, all_end_preds, valid_df):
    invert_maps = valid_df['invert_map'].tolist()
    texts = valid_df['text'].tolist()
    preds = []
    for idx in range(len(texts)):
        text = texts[idx]
        words = text.lower().split()
        invert_map = invert_maps[idx]

        start_word = invert_map[all_start_preds[idx]]
        end_word = invert_map[all_end_preds[idx]]
        selected_text_pred = ' '.join(words[start_word:end_word+1])

        if idx < 20:
            print(selected_text_pred)
        preds.append(selected_text_pred)
    return preds


def get_metrics(all_senti_preds, all_start_preds, all_end_preds, valid_df, args=None):
    all_senti_labels = valid_df['senti_label'].values
    metrics = dict()
    metrics['loss'] = 0
    metrics['senti_acc'] = accuracy_score(all_senti_labels, all_senti_preds)

    invert_maps = valid_df['invert_map'].tolist()
    starts = valid_df['start'].tolist()
    ends = valid_df['end'].tolist()
    texts = valid_df['text'].tolist()
    selected_texts = valid_df['selected_text'].tolist()
    first_tokens = valid_df['first_token'].tolist()
    tokens = valid_df['tokens'].tolist()

    score, score_dirty, score_dirty2 = 0, 0, 0
    for idx in range(len(texts)):
        text = texts[idx]
        words = text.lower().split()
        invert_map = invert_maps[idx]

        if 0<=all_start_preds[idx]<len(invert_map) and 0<=all_end_preds[idx]<len(invert_map) and all_end_preds[idx]>=all_start_preds[idx]:
            start_word = invert_map[all_start_preds[idx]]
            end_word = invert_map[all_end_preds[idx]]+1
            pred = ''
            for pos in range(all_start_preds[idx], all_end_preds[idx]+1):
                if first_tokens[idx][pos]:
                    pred += ' '
                pred += tokens[idx][pos]
        else:
            start_word, end_word = 0, 0
        selected_text_pred = words[start_word:end_word]

        if args and args.post:
            if all_senti_labels[idx]==1:
                selected_text_pred = words

        start_word_label, end_word_label = invert_map[starts[idx]], invert_map[ends[idx]]
        selected_text_label = words[start_word_label: end_word_label+1]
        score += jaccard_list(selected_text_pred, selected_text_label)

        # dirty label & dirty score
        selected_text_label = selected_texts[idx].lower().split()
        score_dirty += jaccard_list(selected_text_pred, selected_text_label)

        score_dirty2 += jaccard_list(pred.lower().split(), selected_text_label)
        if idx < 5:
            print(pred.lower(), '|',
                  ' '.join(selected_text_label))
    metrics['score'] = score/len(texts)
    metrics['dirty_score'] = score_dirty/len(texts)
    metrics['dirty_score2'] = score_dirty2/len(texts)
    print('senti_acc:', metrics['senti_acc'], 
          'jaccard:', metrics['score'], 
          'dirty jaccard:', metrics['dirty_score'],
          'score2', metrics['dirty_score2'])
    print(confusion_matrix(all_senti_labels, all_senti_preds))
    return metrics


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
