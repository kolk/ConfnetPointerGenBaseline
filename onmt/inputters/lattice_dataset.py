# -*- coding: utf-8 -*-
from functools import partial

import six
import torch
import os
from torchtext.data import Field, RawField, NestedField

from onmt.inputters.datareader_base import DataReaderBase


class LatticeDataReader(DataReaderBase):
    def read(self, lattices, side, _dir=""):
        """Read text data from disk.

        Args:
            sequences (str or Iterable[str]):
                Sequence of lattice paths or path to file containing lattice paths.
                text file or iterable of the actual text data.
            side (str): Prefix used in return dict. Usually
                ``"src"`` or ``"tgt"``.
            lattice_dir (NoneType): Leave as ``None``. This parameter exists to
                conform with the :func:`DataReaderBase.read()` signature.

        Yields:
            dictionaries whose keys are the names of fields and whose
            values are more or less the result of tokenizing with those
            fields.
        """

        if isinstance(lattices, str):
            lattices = DataReaderBase._read_file(lattices)

        for i, filename in enumerate(lattices):
            filename = filename.decode("utf-8").strip()
            lattice_path=filename
            #lattice_path = os.path.join(dir, filename)
            if not os.path.exists(lattice_path):
                lattice_path = filename

            assert os.path.exists(lattice_path), \
                'lattice path %s not found' % filename

            text, scores, lens = LatticeDataReader.read_confnet_file(lattice_path)
            yield {side: text, 'score': scores, 'lens': lens, side + '_path': lattice_path, 'indices': i}


    @classmethod
    def read_confnet_file(cls, filename):
        max_par_arc = 20
        max_sent_len = 50
        with open(filename) as f:
            lines = f.readlines()
            lines = lines[3:]
            text = []
            scores = []
            lens = []
            for i, par_arc in enumerate(lines):
                arcs = par_arc.split(' ')[2:]
                text.append(arcs[:max_par_arc:2])
                scores.append([float(j.strip()) for j in arcs[1:max_par_arc:2]])
                lens.append(len(arcs[:max_par_arc:2]))
            #text.append(['</s>'] + ['<blank>']*(max_par_arc-1))
            #scores.append([1.0]+[0.0]*(max_par_arc-1))
            text.extend([['<blank>']*(max_par_arc)]*(max_sent_len-len(text)))
            scores.extend([[0.0]*(max_par_arc)]*(max_sent_len-len(scores)))
            lens.extend([[0]*(max_par_arc)]*(max_sent_len-len(lens)))
            #text.extend(['<blank>'] * (max_sent_len - len(text)))
            #scores.extend([0.0] * (max_sent_len - len(text)))
            #lens.extend([0] * (max_sent_len - len(text)))

        return text, scores, lens

def lattice_sort_key(ex):
    """Sort using the number of tokens in the sequence."""
    if hasattr(ex, "tgt"):
        return len(ex.ques[0]), len(ex.ans[0]), len(ex.tgt[0])
    return len(ex.ques[0]), len(ex.ans[0])


# mix this with partial
def _feature_tokenize(
        string, layer=0, tok_delim=None, feat_delim=None, truncate=None):
    """Split apart word features (like POS/NER tags) from the tokens.

    Args:
        string (str): A string with ``tok_delim`` joining tokens and
            features joined by ``feat_delim``. For example,
            ``"hello|NOUN|'' Earth|NOUN|PLANET"``.
        layer (int): Which feature to extract. (Not used if there are no
            features, indicated by ``feat_delim is None``). In the
            example above, layer 2 is ``'' PLANET``.
        truncate (int or NoneType): Restrict sequences to this length of
            tokens.

    Returns:
        List[str] of tokens.
    """

    tokens = string.split(tok_delim)
    if truncate is not None:
        tokens = tokens[:truncate]
    if feat_delim is not None:
        tokens = [t.split(feat_delim)[layer] for t in tokens]
    return tokens



class LatticeMultiField(RawField):
    """Container for subfields.

    Text data might use POS/NER/etc labels in addition to tokens.
    This class associates the "base" :class:`Field` with any subfields.
    It also handles padding the data and stacking it.

    Args:
        base_name (str): Name for the base field.
        base_field (Field): The token field.
        feats_fields (Iterable[Tuple[str, Field]]): A list of name-field
            pairs.

    Attributes:
        fields (Iterable[Tuple[str, Field]]): A list of name-field pairs.
            The order is defined as the base field first, then
            ``feats_fields`` in alphabetical order.
    """

    def __init__(self, base_name, base_field, feats_fields):
        super(LatticeMultiField, self).__init__()
        self.fields = [(base_name, base_field)]
        for name, ff in sorted(feats_fields, key=lambda kv: kv[0]):
            self.fields.append((name, ff))

    @property
    def base_field(self):
        #print('in base_field')
        #print(self.fields)
        #print(self.fields[0][1].vocab)
        return self.fields[0][1]#, self.fields[1][1]]

    def process(self, batch, device=None):
        """Convert outputs of preprocess into Tensors.

        Args:
            batch (List[List[List[str]]]): A list of length batch size.
                Each element is a list of the preprocess results for each
                field (which are lists of str "words" or feature tags.
            device (torch.device or str): The device on which the tensor(s)
                are built.

        Returns:
            torch.LongTensor or Tuple[LongTensor, LongTensor]:
                A tensor of shape ``(seq_len, batch_size, len(self.fields))``
                where the field features are ordered like ``self.fields``.
                If the base field returns lengths, these are also returned
                and have shape ``(batch_size,)``.
        """

        # batch (list(list(list))): batch_size x len(self.fields) x seq_len
        batch_by_feat = list(zip(*batch))
        #print('base_field')
        #print('base_field dict', self.base_field.__dict__)
        #print(self.base_field.vocab)
        #print('nesting_field dict', self.base_field.nesting_field.__dict__)
        #print(batch_by_feat)
        base_data = self.base_field.process(list(batch_by_feat[0]), device=device)
        if self.base_field.include_lengths:
            # lengths: batch_size
            #print("base_data")
            #print(base_data)
            base_data, sent_lengths, par_arc_lengths = base_data

        """
        feats = [ff.process(batch_by_feat[i], device=device)
                 for i, (_, ff) in enumerate(self.fields[1:], 1)]
        levels = [base_data] + feats
        """
        levels = [base_data]
        # data: seq_len x batch_size x len(self.fields)
        data = torch.stack(levels, 3)
        if self.base_field.include_lengths:
            return data, sent_lengths, par_arc_lengths
        else:
            return data

    def preprocess(self, x):
        """Preprocess data.

        Args:
            x (str): A sentence string (words joined by whitespace).

        Returns:
            List[List[str]]: A list of length ``len(self.fields)`` containing
                lists of tokens/feature tags for the sentence. The output
                is ordered like ``self.fields``.
        """

        return [f.preprocess(x) for _, f in self.fields]

    def __getitem__(self, item):
        return self.fields[item]


def lattice_fields(**kwargs):
    """Create text fields.

    Args:
        base_name (str): Name associated with the field.
        n_feats (int): Number of word level feats (not counting the tokens)
        include_lengths (bool): Optionally return the sequence lengths.
        pad (str, optional): Defaults to ``"<blank>"``.
        bos (str or NoneType, optional): Defaults to ``"<s>"``.
        eos (str or NoneType, optional): Defaults to ``"</s>"``.
        truncate (bool or NoneType, optional): Defaults to ``None``.

    Returns:
        LatticeMultiField
    """

    n_feats = kwargs["n_feats"]
    include_lengths = kwargs["include_lengths"]
    base_name = kwargs["base_name"]
    pad = kwargs.get("pad", "<blank>")
    truncate = kwargs.get("truncate", None)
    fields_ = []
    i = 0
    use_len = i == 0 and include_lengths
    name = base_name + "_feat_" + str(i - 1) if i > 0 else base_name
    nesting_field_text = Field(pad_token=pad, use_vocab=True)
    text_field = NestedField(nesting_field_text, pad_token=pad, include_lengths=use_len, use_vocab=True)

    nesting_field_scores = Field(pad_token=0.0, use_vocab=False, dtype=torch.float64)
    scores_field = NestedField(nesting_field_scores, use_vocab=False, tokenize=None, dtype=torch.float64,
                               include_lengths=use_len)

    #feat = [('confnet', text_field), ('scores', scores_field)]
    fields_.append(('confnet', text_field))
    fields_.append(('scores', scores_field))
    """
    print('fields_', fields_)
    print(fields_[0][0])
    print(fields_[0][1])
    print(fields_[1:])
    """
    #field = LatticeMultiField(fields_[0][0], fields_[0][1], fields_[1:])
    confnet_field = LatticeMultiField('confnet', text_field, [])
    score_field = LatticeMultiField('score', scores_field, [])
    #print('lattice field', field)
    return confnet_field, score_field






