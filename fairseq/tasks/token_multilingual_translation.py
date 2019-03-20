# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import OrderedDict
import os

import torch

from fairseq import options
from fairseq.data import (
    Dictionary, LanguagePairTokenDataset, IndexedDataset, IndexedCachedDataset,
    IndexedRawTextDataset, RoundRobinZipDatasets, AudioDictionary
)
from fairseq.models import FairseqMultiModel

from . import FairseqTask, register_task


@register_task('token_multilingual_translation')
class TokenMultilingualTranslationTask(FairseqTask):
    """A task for training multiple translation models simultaneously.

    We iterate round-robin over batches from multiple language pairs, ordered
    according to the `--lang-pairs` argument.

    The training loop is roughly:

        for i in range(len(epoch)):
            for lang_pair in args.lang_pairs:
                batch = next_batch_for_lang_pair(lang_pair)
                loss = criterion(model_for_lang_pair(lang_pair), batch)
                loss.backward()
            optimizer.step()

    In practice, `next_batch_for_lang_pair` is abstracted in a FairseqDataset
    (e.g., `RoundRobinZipDatasets`) and `model_for_lang_pair` is a model that
    implements the `FairseqMultiModel` interface.

    During inference it is required to specify a single `--source-lang` and
    `--target-lang`, instead of `--lang-pairs`.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language (only needed for inference)')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language (only needed for inference)')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--audio-input', action='store_true',
                            help='load audio input dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left (default: True)')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left (default: False)')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--no-cache-source', default=False, action='store_true')

    def __init__(self, args, dicts, training):
        super().__init__(args)
        self.dicts = dicts
        self.langs = list(dicts.keys())
        self.training = training

    @classmethod
    def setup_task(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if not hasattr(args, 'audio_input'):
            args.audio_input = False

        args.lang_pairs = args.lang_pairs.split(',')
        if args.source_lang is not None or args.target_lang is not None:
            training = False
        else:
            training = True
            args.source_lang, args.target_lang = args.lang_pairs[0].split('-')

        target_langs = sorted(list({lang_pair.split('-')[1] for lang_pair in args.lang_pairs}))
        lang_dict = Dictionary()
        for lang in target_langs:
            lang_dict.add_symbol(TokenMultilingualTranslationTask.make_lang_token(lang))
        lang_dict.finalize()

        # load dictionaries
        dicts = OrderedDict()
        if not args.audio_input:
            dicts['source'] = Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(args.source_lang)))
        else:
            dicts['source'] = AudioDictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(args.source_lang)))
        dicts['target'] = OrderedDict()
        tgt_dict = Dictionary()
        for lang in target_langs:
            dicts['target'][lang] = Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(lang)))
            if len(dicts) > 0:
                assert dicts['target'][lang].pad() == dicts['target'][target_langs[0]].pad()
                assert dicts['target'][lang].eos() == dicts['target'][target_langs[0]].eos()
                assert dicts['target'][lang].unk() == dicts['target'][target_langs[0]].unk()
            print('| [{}] dictionary: {} types'.format(lang, len(dicts['target'][lang])))
            tgt_dict.update(dicts['target'][lang])
        dicts['langs'] = lang_dict
        dicts['tgt_dict'] = tgt_dict

        return cls(args, dicts, training)

    def load_dataset(self, split, **kwargs):
        """Load a dataset split."""

        def split_exists(split, src, tgt, lang):
            filename = os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary, cached=True, audio=False):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedDataset.exists(path):
                if cached:
                    return IndexedCachedDataset(path, fix_lua_indexing=True, audio=audio)
                else:
                    return IndexedDataset(path, fix_lua_indexing=(not audio), audio=audio)
            return None

        def sort_lang_pair(lang_pair):
            return '-'.join(sorted(lang_pair.split('-')))

        if not self.training:
            self.args.lang_pairs = ['{}-{}'.format(self.args.source_lang, self.args.target_lang)]
        src_datasets, tgt_datasets = {}, {}
        for lang_pair in self.args.lang_pairs: #set(map(sort_lang_pair, self.args.lang_pairs)):
            src, tgt = lang_pair.split('-')
            if split_exists(split, src, tgt, src):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
            elif split_exists(split, tgt, src, src):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, tgt, src))
            else:
                continue
            if self.args.audio_input:
                cached = not self.args.no_cache_source
                src_datasets[lang_pair] = indexed_dataset(prefix + src, self.dicts['source'],
                                                          cached=cached, audio=self.args.audio_input)
                tgt_datasets[lang_pair] = indexed_dataset(prefix + tgt, self.dicts['target'][tgt], cached=cached)
            else:
                src_datasets[lang_pair] = indexed_dataset(prefix + src, self.dicts['source'])
                tgt_datasets[lang_pair] = indexed_dataset(prefix + tgt, self.dicts['target'][tgt])
            print('| {} {} {} examples'.format(self.args.data, split, len(src_datasets[lang_pair])))

        if len(src_datasets) == 0:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))


        def define_dict_mapping(source, joint):
            id_map = list()
            for i in range(len(source)):
                word = source[i]
                idx = joint.index(word)
                id_map += [idx]
            return torch.Tensor(id_map).long()

        def language_pair_dataset(lang_pair):
            src, tgt = lang_pair.split('-')
            if lang_pair in src_datasets:
                src_dataset, tgt_dataset = src_datasets[lang_pair], tgt_datasets[lang_pair]
            else:
                lang_pair = sort_lang_pair(lang_pair)
                tgt_dataset, src_dataset = src_datasets[lang_pair], tgt_datasets[lang_pair]
            if self.args.audio_input:
                # saving audio features length, needed when creating the model.
                self.audio_features = src_dataset.sizes[1]
                self.dicts['source'].audio_features = self.audio_features

            lang_token = self.__class__.make_lang_token(tgt)
            lang_id = self.dicts['langs'].index(lang_token)

            return LanguagePairTokenDataset(
                src_dataset, src_dataset.sizes, self.dicts['source'],
                tgt_dataset, tgt_dataset.sizes, self.dicts['target'][tgt],
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
                src_audio=self.args.audio_input,
                lang_id=lang_id, id_map=define_dict_mapping(self.dicts['target'][tgt], self.dicts['tgt_dict'])
            )

        if self.training:
            self.datasets[split] = RoundRobinZipDatasets(
                OrderedDict([
                    (lang_pair, language_pair_dataset(lang_pair))
                    for lang_pair in self.args.lang_pairs
                ]),
                eval_key=None if self.training else self.args.lang_pairs[0],
            )
        else:
            self.datasets[split] = language_pair_dataset(lang_pair)

        self.language_dictionary = self.dicts['langs']

    #def build_model(self, args):
    #    from fairseq import models
    #    model = models.build_model(args, self)
    #    if not isinstance(model, FairseqMultiModel):
    #        raise ValueError('MultilingualTranslationTask requires a FairseqMultiModel architecture')
    #    return model

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
        for lang_pair in self.args.lang_pairs:
            if sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                continue
            loss, sample_size, logging_output = criterion(model, sample[lang_pair])
            if ignore_grad:
                loss *= 0
            optimizer.backward(loss)
            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            agg_logging_output[lang_pair] = logging_output
        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
            for lang_pair in self.args.lang_pairs:
                if sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                    continue
                loss, sample_size, logging_output = criterion(model, sample[lang_pair])
                agg_loss += loss.data.item()
                # TODO make summing of the sample sizes configurable
                agg_sample_size += sample_size
                agg_logging_output[lang_pair] = logging_output
        return agg_loss, agg_sample_size, agg_logging_output

    def init_logging_output(self, sample):
        return {
            'ntokens': sum(
                sample_lang.get('ntokens', 0)
                for sample_lang in sample.values()
            ) if sample is not None else 0,
            'nsentences': sum(
                sample_lang['target'].size(0) if 'target' in sample_lang else 0
                for sample_lang in sample.values()
            ) if sample is not None else 0,
        }

    def grad_denom(self, sample_sizes, criterion):
        return criterion.__class__.grad_denom(sample_sizes)

    def aggregate_logging_outputs(self, logging_outputs, criterion):
        # aggregate logging outputs for each language pair
        agg_logging_outputs = {
            lang_pair: criterion.__class__.aggregate_logging_outputs([
                logging_output.get(lang_pair, {}) for logging_output in logging_outputs
            ])
            for lang_pair in self.args.lang_pairs
        }

        def sum_over_languages(key):
            return sum(logging_output[key] for logging_output in agg_logging_outputs.values())

        # flatten logging outputs
        flat_logging_output = {
            '{}:{}'.format(lang_pair, k): v
            for lang_pair, agg_logging_output in agg_logging_outputs.items()
            for k, v in agg_logging_output.items()
        }
        flat_logging_output['loss'] = sum_over_languages('loss')
        flat_logging_output['nll_loss'] = sum_over_languages('nll_loss')
        flat_logging_output['sample_size'] = sum_over_languages('sample_size')
        flat_logging_output['nsentences'] = sum_over_languages('nsentences')
        flat_logging_output['ntokens'] = sum_over_languages('ntokens')
        return flat_logging_output

    @property
    def source_dictionary(self):
        return self.dicts['source']

    @property
    def target_dictionary(self):
        return self.dicts['tgt_dict']

    @staticmethod
    def make_lang_token(lang):
        return '<2' + lang + '>'

    def max_positions(self):
        """Return the max input length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions) if not self.training else None
        #return OrderedDict(
        #    {key: (self.args.max_source_positions, self.args.max_target_positions)
        #    for key in self.args.lang_pairs
        #    })
