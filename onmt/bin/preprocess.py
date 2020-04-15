#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""
import codecs
import glob
import gc
import torch
from collections import Counter, defaultdict

from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import split_corpus
import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.inputters.inputter import _build_fields_vocab,\
                                    _load_vocab

from functools import partial
from multiprocessing import Pool


def check_existing_pt_files(opt, corpus_type, ids, existing_fields):
    """ Check if there are existing .pt files to avoid overwriting them """
    existing_shards = []
    for maybe_id in ids:
        if maybe_id:
            shard_base = corpus_type + "_" + maybe_id
        else:
            shard_base = corpus_type
        pattern = opt.save_data + '.{}.*.pt'.format(shard_base)
        if glob.glob(pattern):
            if opt.overwrite:
                maybe_overwrite = ("will be overwritten because "
                                   "`-overwrite` option is set.")
            else:
                maybe_overwrite = ("won't be overwritten, pass the "
                                   "`-overwrite` option if you want to.")
            logger.warning("Shards for corpus {} already exist, {}"
                           .format(shard_base, maybe_overwrite))
            existing_shards += [maybe_id]
    return existing_shards


def process_one_shard(corpus_params, params):
    #print('corpus_params', corpus_params)
    #print('params', params)
    corpus_type, fields, ques_reader, ans_reader, tgt_reader, align_reader, opt,\
         existing_fields, src_vocab, tgt_vocab, confnet_vocab = corpus_params
    i, (ques_shard, ans_shard, tgt_shard, align_shard, maybe_id, filter_pred) = params
    print('in process_one_shard ')

    # create one counter per shard
    sub_sub_counter = defaultdict(Counter)
    #print('sub_sub_counter', sub_sub_counter)
    logger.info("Building shard %d." % i)

    ans_data = {"reader": ans_reader, "data": ans_shard, "dir": opt.src_dir}
    tgt_data = {"reader": tgt_reader, "data": tgt_shard, "dir": None}
    ques_data = {"reader": ques_reader, "data": ques_shard, "dir":None}
    align_data = {"reader": align_reader, "data": align_shard, "dir": None}
    print('all_data_done')
    _readers, _data, _dir = inputters.Dataset.config(
        [('ques', ques_data), ('ans', ans_data), ('tgt', tgt_data), ('align', align_data)])

    print('Dataset.config done')
    dataset = inputters.Dataset(
        fields, readers=_readers, data=_data, dirs=_dir,
        sort_key=inputters.str2sortkey[opt.data_type],
        filter_pred=filter_pred
    )
    #print('inputters.dataset done')
    #print('dataset', dataset.__dict__)
    if corpus_type == "train" and existing_fields is None:
        for ex in dataset.examples:
            for name, field in fields.items():
                #print('ex', ex)
                #print('field', field)
                if ((opt.data_type == "audio") and (name == "src")):
                    continue
                try:
                    f_iter = iter(field)
                except TypeError:
                    f_iter = [(name, field)]
                    all_data = [getattr(ex, name, None)]
                else:
                    all_data = getattr(ex, name)
                    #print('all_data', all_data)
                for (sub_n, sub_f), fd in zip(
                        f_iter, all_data):
                    has_vocab = (sub_n == 'ans' and
                                 src_vocab is not None) or \
                                (sub_n == 'tgt' and
                                 tgt_vocab is not None) or \
                                (sub_n == 'ques' and
                                 confnet_vocab is not None)
                    if (hasattr(sub_f, 'sequential')
                            and sub_f.sequential and not has_vocab):
                        if sub_n == "score":
                            continue
                        if sub_n == 'confnet':
                            val = [w for par_arc in fd for w in par_arc]
                        else:
                            val = fd
                        sub_sub_counter[sub_n].update(val)
    if maybe_id:
        shard_base = corpus_type + "_" + maybe_id
    else:
        shard_base = corpus_type
    data_path = "{:s}.{:s}.{:d}.pt".\
        format(opt.save_data, shard_base, i)

    logger.info(" * saving %sth %s data shard to %s."
                % (i, shard_base, data_path))

    dataset.save(data_path)

    del dataset.examples
    gc.collect()
    del dataset
    gc.collect()

    return sub_sub_counter


def maybe_load_vocab(corpus_type, counters, opt):
    src_vocab = None
    tgt_vocab = None
    confnet_vocab = None
    existing_fields = None
    if corpus_type == "train":
        if opt.src_vocab != "":
            try:
                logger.info("Using existing vocabulary...")
                existing_fields = torch.load(opt.src_vocab)
            except torch.serialization.pickle.UnpicklingError:
                logger.info("Building vocab from text file...")
                src_vocab, src_vocab_size = _load_vocab(
                    opt.src_vocab, "src", counters,
                    opt.src_words_min_frequency)
        if opt.tgt_vocab != "":
            tgt_vocab, tgt_vocab_size = _load_vocab(
                opt.tgt_vocab, "tgt", counters,
                opt.tgt_words_min_frequency)

        if opt.confnet_vocab != "":
            confnet_vocab, confnet_vocab_size = _load_vocab(
                opt.confnet_vocab, "confnet", counters,
                opt.confnet_words_min_frequency)
    return confnet_vocab, src_vocab, tgt_vocab, existing_fields


def build_save_dataset(corpus_type, fields, ques_reader, ans_reader, tgt_reader,
                       align_reader, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        counters = defaultdict(Counter)
        ans = opt.train_src
        ques = opt.train_confnet
        tgts = opt.train_tgt
        ids = opt.train_ids
        aligns = opt.train_align
    elif corpus_type == 'valid':
        counters = None
        ans = [opt.valid_src]
        ques = [opt.valid_confnet]
        tgts = [opt.valid_tgt]
        ids = [None]
        aligns = [opt.valid_align]
    print('before maybe_load_vocab')
    ques_vocab, ans_vocab, tgt_vocab, existing_fields = maybe_load_vocab(
        corpus_type, counters, opt)
    print('after maybe_load_vocab')
    existing_shards = check_existing_pt_files(
        opt, corpus_type, ids, existing_fields)

    # every corpus has shards, no new one
    if existing_shards == ids and not opt.overwrite:
        return

    def shard_iterator(ques, ans, tgts, ids, aligns, existing_shards,
                       existing_fields, corpus_type, opt):
        """
        Builds a single iterator yielding every shard of every corpus.
        """
        for q, a, tgt, maybe_id, maybe_align in zip(ques, ans, tgts, ids, aligns):
            if maybe_id in existing_shards:
                if opt.overwrite:
                    logger.warning("Overwrite shards for corpus {}"
                                   .format(maybe_id))
                else:
                    if corpus_type == "train":
                        assert existing_fields is not None,\
                            ("A 'vocab.pt' file should be passed to "
                             "`-src_vocab` when adding a corpus to "
                             "a set of already existing shards.")
                    logger.warning("Ignore corpus {} because "
                                   "shards already exist"
                                   .format(maybe_id))
                    continue
            if ((corpus_type == "train" or opt.filter_valid)
                    and tgt is not None):

                filter_pred = partial(
                    inputters.filter_example,
                    use_src_len=opt.data_type == "lattice",
                    use_confnet_len=opt.data_type == "lattice",
                    max_src_len=opt.src_seq_length,
                    max_tgt_len=opt.tgt_seq_length)
            else:
                filter_pred = None

            ans_shards = split_corpus(a, opt.shard_size)
            tgt_shards = split_corpus(tgt, opt.shard_size)
            align_shards = split_corpus(maybe_align, opt.shard_size)
            ques_shards = split_corpus(q, opt.shard_size)

            for i, (qs, anss, ts, a_s) in enumerate(
                    zip(ques_shards, ans_shards, tgt_shards, align_shards)):
                #print('qs', qs)
                #print('ans', anss)
                #print('tgt', ts)
                yield (i, (qs, anss, ts, a_s, maybe_id, filter_pred))

    shard_iter = shard_iterator(ques, ans, tgts, ids, aligns, existing_shards,
                                existing_fields, corpus_type, opt)
    #print('after shard_iter')
    with Pool(opt.num_threads) as p:
        dataset_params = (corpus_type, fields, ques_reader, ans_reader, tgt_reader,
                          align_reader, opt, existing_fields,
                          ans_vocab, tgt_vocab, ques_vocab)
        #print('dataset_params', dataset_params)

        func = partial(process_one_shard, dataset_params)
        for sub_counter in p.imap(func, shard_iter):
            if sub_counter is not None:
                for key, value in sub_counter.items():
                    counters[key].update(value)


    if corpus_type == "train":
        vocab_path = opt.save_data + '.vocab.pt'
        if existing_fields is None:
            fields = _build_fields_vocab(
                fields, counters, opt.data_type,
                opt.share_vocab, opt.vocab_size_multiple,
                opt.confnet_vocab_size, opt.confnet_words_min_frequency,
                opt.src_vocab_size, opt.src_words_min_frequency,
                opt.tgt_vocab_size, opt.tgt_words_min_frequency)
        else:
            fields = existing_fields
        torch.save(fields, vocab_path)


def build_save_vocab(train_dataset, fields, opt):
    fields = inputters.build_vocab(
        train_dataset, fields, opt.data_type, opt.share_vocab,
        opt.confnet_vocab_size, opt.confnet_words_min_frequency,
        opt.src_vocab, opt.src_vocab_size, opt.src_words_min_frequency,
        opt.tgt_vocab, opt.tgt_vocab_size, opt.tgt_words_min_frequency,
        vocab_size_multiple=opt.vocab_size_multiple
    )
    vocab_path = opt.save_data + '.vocab.pt'
    torch.save(fields, vocab_path)


def count_features(path):
    """
    path: location of a corpus file with whitespace-delimited tokens and
                    ￨-delimited features within the token
    returns: the number of features in the dataset
    """
    with codecs.open(path, "r", "utf-8") as f:
        first_tok = f.readline().split(None, 1)[0]
        return len(first_tok.split(u"￨")) - 1


def preprocess(opt):
    ArgumentParser.validate_preprocess_args(opt)
    torch.manual_seed(opt.seed)

    init_logger(opt.log_file)

    logger.info("Extracting features...")

    src_nfeats = 0
    tgt_nfeats = 0
    confnet_nfeats = 0
    for src, tgt, cnet in zip(opt.train_src, opt.train_tgt, opt.train_confnet):
        src_nfeats += count_features(src) if opt.data_type == 'text' or opt.data_type == 'lattice' \
            else 0
        tgt_nfeats += count_features(tgt)  # tgt always text so far
        #confnet_nfeats += count_features(cnet) if opt.data_type == 'lattice' \
        #    else 0
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target features: %d." % tgt_nfeats)
    logger.info(" * number of confnet features: %d." % confnet_nfeats)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(
        opt.data_type,
        src_nfeats,
        confnet_nfeats,
        tgt_nfeats,
        dynamic_dict=opt.dynamic_dict,
        with_align=opt.train_align[0] is not None,
        ans_truncate=opt.src_seq_length_trunc,
        ques_truncate=opt.confnet_seq_length_trunc,
        tgt_truncate=opt.tgt_seq_length_trunc)
    #print('fields done')
    ans_reader = inputters.str2reader["text"].from_opt(opt)
    tgt_reader = inputters.str2reader["text"].from_opt(opt)
    align_reader = inputters.str2reader["text"].from_opt(opt)
    ques_reader = inputters.str2reader["lattice"].from_opt(opt)
    #print('src_reader', ques_reader)
    #print('tgt_reader', tgt_reader)
    #print('aglign_reader', align_reader)
    #print('confnet_reader', ans_reader)
    logger.info("Building & saving training data...")
    build_save_dataset(
        'train', fields, ques_reader, ans_reader, tgt_reader, align_reader, opt)

    if opt.valid_src and opt.valid_tgt:
        logger.info("Building & saving validation data...")
        build_save_dataset(
            'valid', fields, ques_reader, ans_reader, tgt_reader, align_reader, opt)


def _get_parser():
    parser = ArgumentParser(description='preprocess.py')

    opts.config_opts(parser)
    opts.preprocess_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    preprocess(opt)


if __name__ == "__main__":
    main()
