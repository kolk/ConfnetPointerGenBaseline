# coding: utf-8

from itertools import chain, starmap
from collections import Counter

import torch
from torchtext.data import Dataset as TorchtextDataset
from torchtext.data import Example
from torchtext.vocab import Vocab


def _join_dicts(*args):
    """
    Args:
        dictionaries with disjoint keys.

    Returns:
        a single dictionary that has the union of these keys.
    """

    return dict(chain(*[d.items() for d in args]))


def _dynamic_dict(example, ques_field, ans_field, tgt_field, max_par_arc_size=20):
    """Create copy-vocab and numericalize with it.

    In-place adds ``"src_map"`` to ``example``. That is the copy-vocab
    numericalization of the tokenized ``example["src"]``. If ``example``
    has a ``"tgt"`` key, adds ``"alignment"`` to example. That is the
    copy-vocab numericalization of the tokenized ``example["tgt"]``. The
    alignment has an initial and final UNK token to match the BOS and EOS
    tokens.

    Args:
        example (dict): An example dictionary with a ``"src"`` key and
            maybe a ``"tgt"`` key. (This argument changes in place!)
        src_field (torchtext.data.Field): Field object.
        tgt_field (torchtext.data.Field): Field object.

    Returns:
        torchtext.data.Vocab and ``example``, changed as described.
    """
    #print(example.keys())
    #print('ans', example["ans"])
    #print('ques', example["ques"])
    ans = ans_field.tokenize(example["ans"])
    if isinstance(example["ques"], str):
        ques = ques_field.tokenize(example["ques"])
    else:
        # confnet
        ques = example["ques"]
        ques_weights = example["score"]
        #example["ques"] = [ques, ques_weights]

    if isinstance(example["ans"], str):
        ans = ans_field.tokenize(example["ans"])
    else:
        # confnet
        ans = example["ans"]

    # make a small vocab containing just the tokens in the source sequence
    unk = ans_field.unk_token
    pad = ans_field.pad_token
    assert unk == ques_field.unk_token
    assert pad == ques_field.pad_token

    """
    if isinstance(example["ans"], str):
        src_ex_vocab = Vocab(Counter([w for par_arcs in ques for w in par_arcs ] + ans), specials=[unk, pad])
        ans_map = [src_ex_vocab.stoi[w] for w in ans]
    else:
        src_count = Counter([w for par_arcs in ques for w in par_arcs] + [w for par_arcs in ans for w in par_arcs])
        src_ex_vocab = Vocab(src_count, specials=[unk, pad])
        ans_map = torch.LongTensor([[src_ex_vocab.stoi[w] for w in par_arcs] for par_arcs in ques])

    if isinstance(example["ques"], str):
        if isinstance(example["ans"], str):
            src_ex_vocab = Vocab(Counter(ques+ans), specials=[unk, pad])
            ques_map = [src_ex_vocab.stoi[w] for w in ques]
            ans_map = [src_ex_vocab.stoi[w] for w in ques]

        elif isinstance(example["ans"], list):
            src_count = Counter(ques + [w for par_arcs in ans for w in par_arcs])
            src_ex_vocab = Vocab(src_count, specials=[unk, pad])

            ques_map = [[src_ex_vocab.stoi[w]]+[pad]*(max_par_arc_size-1) for w in ques]
            ans_map = [[src_ex_vocab.stoi[w] for w in par_arcs]+[pad]*(max_par_arc_size-len(par_arcs)) for par_arcs in ans]

    elif isinstance(example["ques"], list):
        if isinstance(example["ans"], str):
            src_count = Counter(ans + [w for par_arcs in ques for w in par_arcs])
            src_ex_vocab = Vocab(src_count, specials=[unk, pad])
            temp_map = [src_ex_vocab.stoi[w] for w in ans]
            ans_map = [[src_ex_vocab.stoi[w]] + [src_ex_vocab.stoi[pad]] * (max_par_arc_size - 1) for w in ans]
            ques_map = [[src_ex_vocab.stoi[w] for w in par_arcs] + [src_ex_vocab.stoi[pad]] * (max_par_arc_size - len(par_arcs)) for
                       par_arcs in ques]

            ans_map_weights = [[1.0] + [0] * (max_par_arc_size - 1) for w in ans]
            ques_map_weights =  [[w for w in par_arcs] + [0] * (max_par_arc_size - len(par_arcs)) for
                       par_arcs in ques_weights]
    
        elif isinstance(example["ans"], list):
            src_count = Counter([w for par_arcs in ques for w in par_arcs] + [w for par_arcs in ans for w in par_arcs])
            src_ex_vocab = Vocab(src_count, specials=[unk, pad])
            ans_map = [[src_ex_vocab.stoi[w] for w in par_arcs] + [pad] * (max_par_arc_size - len(par_arcs)) for par_arcs in ans]
            ques_map = [[src_ex_vocab.stoi[w] for w in par_arcs] + [pad] * (max_par_arc_size - len(par_arcs)) for
                        par_arcs in ques]
    """

    src_count = Counter(ans + [w for par_arcs in ques for w in par_arcs])
    src_ex_vocab = Vocab(src_count, specials=[unk, pad])
    ans_map = [[src_ex_vocab.stoi[w]] + [src_ex_vocab.stoi[pad]] * (max_par_arc_size - 1) for w in ans]
    ques_map = [[src_ex_vocab.stoi[w] for w in par_arcs] + [src_ex_vocab.stoi[pad]] * (max_par_arc_size - len(par_arcs))
                for par_arcs in ques]

    ans_map_weights = [[1.0] + [0] * (max_par_arc_size - 1) for w in ans]
    ques_map_weights = [[w for w in par_arcs] + [0] * (max_par_arc_size - len(par_arcs)) for
                        par_arcs in ques_weights]

    unk_idx = src_ex_vocab.stoi[unk]
    # Map source tokens to indices in the dynamic dict.
    src_map = torch.cat((torch.LongTensor(ques_map), torch.LongTensor(ans_map)), dim=0)
    src_map_weights = torch.cat((torch.FloatTensor(ques_map_weights), torch.FloatTensor(ans_map_weights)), dim=0)
    example["src_map"] = [src_map, src_map_weights]
    example["src_ex_vocab"] = src_ex_vocab

    if "tgt" in example:
        tgt = tgt_field.tokenize(example["tgt"])
        mask = torch.LongTensor(
            [unk_idx] + [src_ex_vocab.stoi[w] for w in tgt] + [unk_idx])
        example["alignment"] = mask
    return src_ex_vocab, example


class Dataset(TorchtextDataset):
    """Contain data and process it.

    A dataset is an object that accepts sequences of raw data (sentence pairs
    in the case of machine translation) and fields which describe how this
    raw data should be processed to produce tensors. When a dataset is
    instantiated, it applies the fields' preprocessing pipeline (but not
    the bit that numericalizes it or turns it into batch tensors) to the raw
    data, producing a list of :class:`torchtext.data.Example` objects.
    torchtext's iterators then know how to use these examples to make batches.

    Args:
        fields (dict[str, Field]): a dict with the structure
            returned by :func:`onmt.inputters.get_fields()`. Usually
            that means the dataset side, ``"src"`` or ``"tgt"``. Keys match
            the keys of items yielded by the ``readers``, while values
            are lists of (name, Field) pairs. An attribute with this
            name will be created for each :class:`torchtext.data.Example`
            object and its value will be the result of applying the Field
            to the data that matches the key. The advantage of having
            sequences of fields for each piece of raw input is that it allows
            the dataset to store multiple "views" of each input, which allows
            for easy implementation of token-level features, mixed word-
            and character-level models, and so on. (See also
            :class:`onmt.inputters.TextMultiField`.)
        readers (Iterable[onmt.inputters.DataReaderBase]): Reader objects
            for disk-to-dict. The yielded dicts are then processed
            according to ``fields``.
        data (Iterable[Tuple[str, Any]]): (name, ``data_arg``) pairs
            where ``data_arg`` is passed to the ``read()`` method of the
            reader in ``readers`` at that position. (See the reader object for
            details on the ``Any`` type.)
        dirs (Iterable[str or NoneType]): A list of directories where
            data is contained. See the reader object for more details.
        sort_key (Callable[[torchtext.data.Example], Any]): A function
            for determining the value on which data is sorted (i.e. length).
        filter_pred (Callable[[torchtext.data.Example], bool]): A function
            that accepts Example objects and returns a boolean value
            indicating whether to include that example in the dataset.

    Attributes:
        src_vocabs (List[torchtext.data.Vocab]): Used with dynamic dict/copy
            attention. There is a very short vocab for each src example.
            It contains just the source words, e.g. so that the generator can
            predict to copy them.
    """

    def __init__(self, fields, readers, data, dirs, sort_key,
                 filter_pred=None):
        #print('fields', fields)
        #print('readers', readers)
        #print('dirs', dirs)
        self.sort_key = sort_key
        can_copy = 'src_map' in fields and 'alignment' in fields

        #print('can_copy', can_copy)
        read_iters = [r.read(dat[1], dat[0], dir_) for r, dat, dir_
                      in zip(readers, data, dirs)]

        # self.src_vocabs is used in collapse_copy_scores and Translator.py
        self.src_vocabs = []
        examples = []
        for ex_dict in starmap(_join_dicts, zip(*read_iters)):
            #print('qields ques', ex_dict)
            if can_copy:
                ques_field = fields['ques']
                ans_field = fields['ans']
                tgt_field = fields['tgt']
                # this assumes src_field and tgt_field are both text
                src_ex_vocab, ex_dict = _dynamic_dict(
                    ex_dict, ques_field.base_field, ans_field.base_field, tgt_field.base_field)
                self.src_vocabs.append(src_ex_vocab)

            ex_fields = {k: [(k, v)] for k, v in fields.items() if
                         k in ex_dict}

            ########## HACK ##############
            #ex_dict["ques"] = [ex_dict["ques"], ex_dict["scores"]]
            #############################
            #print(ex_fields)
            #print(ex_dict)
            #ex_fields["src_map_weights"] = ex_dict["src_map_weights"]
            ex = Example.fromdict(ex_dict, ex_fields)
            #scores_dict = {k: [(k, v)] for k, v in fields.items() if k == "scores"}
            #ex_temp = Example.fromdict(scores_dict, ex_fields)
            #print(ex.ques)
            #print(ex.ans)
            examples.append(ex)

        # fields needs to have only keys that examples have as attrs
        fields = []
        for _, nf_list in ex_fields.items():
            assert len(nf_list) == 1
            fields.append(nf_list[0])

        super(Dataset, self).__init__(examples, fields, filter_pred)

    def __getattr__(self, attr):
        # avoid infinite recursion when fields isn't defined
        if 'fields' not in vars(self):
            raise AttributeError
        if attr in self.fields:
            return (getattr(x, attr) for x in self.examples)
        else:
            raise AttributeError

    def save(self, path, remove_fields=True):
        if remove_fields:
            self.fields = []
        torch.save(self, path)

    @staticmethod
    def config(fields):
        readers, data, dirs = [], [], []
        for name, field in fields:
            if field["data"] is not None:
                readers.append(field["reader"])
                data.append((name, field["data"]))
                dirs.append(field["dir"])
        return readers, data, dirs
