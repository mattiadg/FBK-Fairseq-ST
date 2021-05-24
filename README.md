# FBK-fairseq-ST

FBK-fairseq-ST is an adaptation of [FAIR's fairseq](https://github.com/pytorch/fairseq) for direct speech translation.
This repo is no longer active, you should refer to [this fork](https://github.com/mgaido91/FBK-fairseq-ST) under development.

This software has been used for the experiments of the following publications:
* [Fine-tuning on Clean Data for End-to-End Speech Translation: FBK@ IWSLT 2018](https://arxiv.org/abs/1810.07652)
* [MuST-C: a Multilingual Speech Translation Corpus](https://www.aclweb.org/anthology/N19-1202)
* [Enhancing Transformer for End-to-end Speech-to-Text Translation](https://docs.wixstatic.com/ugd/705d57_e6b5a5c517fc41769bdd57b67e57bdc9.pdf)

It also implements the speech translation model proposed in [End-to-End Automatic Speech Translation of Audiobooks
](https://arxiv.org/abs/1802.04200) and the Gaussian distance penalty introduced in [Self-Attentional Acoustic Models
](https://arxiv.org/abs/1803.09519).

The pre-trained models for those papers can be found [here](https://ict.fbk.eu/st-fairseq-models/) and the respective dictionaries can be found [here](https://github.com/mattiadg/FBK-Fairseq-ST/examples/speech_translation/dictionaries).

At the [bottom](#Introduction) of this file you can find the official documentation of this fairseq-py version.


## Requirements and Installation
* A [PyTorch installation](http://pytorch.org/)
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version >= 0.4.0 

Please follow the instructions here: https://github.com/pytorch/pytorch#installation.


After PyTorch is installed, you can install fairseq with:

```
git clone git@github.com:mattiadg/FBK-Fairseq-ST.git
cd <path to downloaded FBK-fairseq repository>

pip install -r requirements.txt
python setup.py build develop
```

## Preprocessing: Data preparation
To reproduce our experiments, the textual side of data should be first tokenized, then split in characters. For tokenization we used the [Moses scripts](https://github.com/moses-smt/mosesdecoder/tree/master/scripts):
```
$moses_scripts/tokenizer/tokenizer.perl -l $LANG < $INPUT_FILE | $moses_scripts/tokenizer/deescape-special-chars.perl > $INPUT_FILE.tok
bash FBK-fairseq-st/scripts/word_level2char_level.sh $INPUT_FILE.tok
```

## Preprocessing: binarizing data

As of now, the only supported audio format are *.npz* and *.h5*.

```
python preprocess.py -s <source_language> -t <target_language> --format <h5 | npz> --inputtype audio \
	--trainpref <path_to_train_data> [[--validpref <path_to_validation_data>] \
	[--testpref <path_to_test_data>]] --destdir <path to output folder>
```

Remember that the input/output dataset must have the same name and be in the same folder.
e.g. if you want to binarize IWSLT en-de data in foo/bar/train_iwslt an example of file structure is as follows:

```
foo
|---- bar
	|----- train_iwstlt
			  |---- my_data.npz
			  |---- my_data.de
```

In this example `--trainpref` should then be foo/bar/train_iwslt/my_data.
The same holds for `--validpref` and `--testpref` i.e. the test and valid (dev) sets could be in different folders but the name of audio related text must be the same for every dataset split.

## Training a new model

This is the minimum required command to train a seq2seq model for ST. 

*NOTE*: training on cpu is not supported, `CUDA_VISIBLE_DEVICES` must be set (else all available gpus will be used).

```
python train.py <path_to folder_with_binary_data> \
	--arch {fconv | transformer | ast_seq2seq | ber2transf | ecc} \
	--save-dir <path_where_to_store_model_checkpoints> --task translation --audio-input
```


The path to the binarized data should point to the FOLDER, e.g. if your data is in *foo/bar/{train-en-de.bin, train-en-de.idx, etc, etc}* then <path_to folder_with_binary_data> should be foo/bar.


Specific architecture variants can be used by setting e.g. `--arch transformer_iwslt_fbk` is a variant of the transformer created by us the differences are in number of layers, dropout value ecc..
	
```
available architecures:
			transformer, transformer_iwslt_fbk,
			transformer_iwslt_de_en, transformer_wmt_en_de,
			transformer_vaswani_wmt_en_de_big,
			transformer_vaswani_wmt_en_fr_big,
			transformer_wmt_en_de_big,
			transformer_wmt_en_de_big_t2t, ast_seq2seq,
			ber2transf, transf2ber,fconv,
			fconv_iwslt_de_en, fconv_wmt_en_ro,
			fconv_wmt_en_de, fconv_wmt_en_fr (default: fconv)
```

The architecrure used for IWSLT2018 is ast_seq2seq.
To reproduce our IWSLT2018 result on the IWSLT en-de Ted Corpus the following command should be used (please substitute the bracketed commands accordingly:


```
CUDA_VISIBLE_DEVICES=[gpu id] python train.py [path to binarized IWSLT data] \
    --clip-norm 5 \
    --max-sentences 32 \
    --max-tokens 100000 \
    --save-dir [output folder] \
    --max-epoch 150 \
    --lr 0.001 \
    --lr-shrink 1.0 \
    --min-lr 1e-08 \
    --dropout 0.2 \
    --lr-schedule fixed \
    --optimizer adam \
    --arch ast_seq2seq \
    --decoder-attention True \
    --seed 666 \
    --task translation \
    --skip-invalid-size-inputs-valid-test \
    --sentence-avg \
    --attention-type general \
    --learn-initial-state \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1
```

To reproduce the results on MuST-C of the paper "Adapting Transformer to End-to-End Spoken Language Translation" run the following (on 4 gpus):
```

CUDA_VISIBLE_DEVICES=[gpu id] python train.py [path to binarized MuST-C data] \
    --clip-norm 20 \
    --max-sentences 8 \
    --max-tokens 12000 \
    --save-dir [output folder] \
    --max-epoch 100 \
    --lr 5e-3 \
    --min-lr 1e-08 \
    --dropout 0.1 \
    --lr-schedule inverse_sqrt \
    --warmup-updates 4000 --warmup-init-lr 3e-4 \
    --optimizer adam \
    --arch speechconvtransformer_big \
    --task translation \
    --audio-input \
    --max-source-positions 1400 --max-target-positions 300 \
    --update-freq 16 \
    --skip-invalid-size-inputs-valid-test \
    --sentence-avg \
    --distance-penalty {log,gauss} \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1
```



Architecture-specific parameter can be specified through command line arguments (highest priority) or through code. Code must be added at the end of a model file in *fairseq/models/* using the `@register_model_architecture` decorator.
The possibile parameters to be changed can be found in every model file in the `add_args` method.


## Generation: translating audio

```
python generate.py <path_to_binarized_data_FOLDER> --path \
	<path_to_checkpoint_to_use> --task translation --audio-input\
	[[--gen-subset valid] [--beam 5] [--batch 32] \
	[--quiet] [--skip-invalid-size-inputs-valid-test]] \
	
```

With the `--quiet` flag only the translations (i.e. hypothesis of the model) will be printed on stdout, no probabilities or reference translations will be shown. Remove the `--quiet` flag for a more verbose output.


*NOTE*: translations are generated following a length order criterion (shortest samples first). Thus the order is not the same as the one in the origianal [test | dev ] set. This is an optimization trick done by fairseq at generation time. Thus the original reference translation are usually not a good solution to compute the BLEU score. A specific reference translation file must be used. It is IMPORTANT to note that said reference translation file is dependent on the `--batch value` used to generate the hypothesis with the system. This happens because the length order used to output the translation is also dependent on such value.
This means a specific reference translation file for every `--batch` value used must be created. A reference file can be created by generating the translations without `--quiet` flag, redirecting stdout to a file and then pass such file as input to the script *sort-sentences.py*, then bring it back to words:
```
python sort-sentences.py $TRANSLATION 5 > $TRANSLATION.sort
sh extract_words.sh $TRANSLATION.sort
```


For every other aspects please refer to the official [fairseq-py](https://github.com/pytorch/fairseq/blob/master/README.md) documentation.


*Note:* Fairseq-py official documentation does not include audio processing and it could change to track fairseq official development, thus the official documentation *could* be incompatible with our version.

## Citation

If you use this software for your research, then please cite it as:
```
@article{di2019adapting,
  title={Adapting Transformer to End-to-End Spoken Language Translation},
  author={Di Gangi, Mattia A and Negri, Matteo and Turchi, Marco},
  journal={Proc. Interspeech 2019},
  pages={1133--1137},
  year={2019}
}
```

## Acknowledgment
This codebase is part of a project financially supported by an Amazon ML Grant.

======================================

The following was the official fairseq-py documentation when we began developing FBK-fairseq (August 2018)

# Introduction

Fairseq(-py) is a sequence modeling toolkit that allows researchers and
developers to train custom models for translation, summarization, language
modeling and other text generation tasks. It provides reference implementations
of various sequence-to-sequence models, including:
- **Convolutional Neural Networks (CNN)**
  - [Dauphin et al. (2017): Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083)
  - [Gehring et al. (2017): Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)
  - [Edunov et al. (2018): Classical Structured Prediction Losses for Sequence to Sequence Learning](https://arxiv.org/abs/1711.04956)
  - **_New_** [Fan et al. (2018): Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833)
- **Long Short-Term Memory (LSTM) networks**
  - [Luong et al. (2015): Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
  - [Wiseman and Rush (2016): Sequence-to-Sequence Learning as Beam-Search Optimization](https://arxiv.org/abs/1606.02960)
- **Transformer (self-attention) networks**
  - [Vaswani et al. (2017): Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - **_New_** [Ott et al. (2018): Scaling Neural Machine Translation](https://arxiv.org/abs/1806.00187)
  - **_New_** [Edunov et al. (2018): Understanding Back-Translation at Scale](https://arxiv.org/abs/1808.09381)

Fairseq features:
- multi-GPU (distributed) training on one machine or across multiple machines
- fast beam search generation on both CPU and GP
- large mini-batch training even on a single GPU via delayed updates
- fast half-precision floating point (FP16) training
- extensible: easily register new models, criterions, and tasks

We also provide [pre-trained models](#pre-trained-models) for several benchmark
translation and language modeling datasets.

![Model](fairseq.gif)

# Requirements and Installation
* A [PyTorch installation](http://pytorch.org/)
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.6

Currently fairseq requires PyTorch version >= 0.4.0.
Please follow the instructions here: https://github.com/pytorch/pytorch#installation.

If you use Docker make sure to increase the shared memory size either with
`--ipc=host` or `--shm-size` as command line options to `nvidia-docker run`.

After PyTorch is installed, you can install fairseq with:
```
pip install -r requirements.txt
python setup.py build develop
```

# Getting Started

The [full documentation](https://fairseq.readthedocs.io/) contains instructions
for getting started, training new models and extending fairseq with new model
types and tasks.

# Pre-trained Models

We provide the following pre-trained models and pre-processed, binarized test sets:

### Translation

Description | Dataset | Model | Test set(s)
---|---|---|---
Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/models/wmt14.v2.en-fr.fconv-py.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/data/wmt14.v2.en-fr.newstest2014.tar.bz2) <br> newstest2012/2013: <br> [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/data/wmt14.v2.en-fr.ntst1213.tar.bz2)
Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT14 English-German](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/models/wmt14.en-de.fconv-py.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/data/wmt14.en-de.newstest2014.tar.bz2)
Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT17 English-German](http://statmt.org/wmt17/translation-task.html#Download) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/models/wmt17.v2.en-de.fconv-py.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/data/wmt17.v2.en-de.newstest2014.tar.bz2)
Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/models/wmt14.en-fr.joined-dict.transformer.tar.bz2) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/data/wmt14.en-fr.joined-dict.newstest2014.tar.bz2)
Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT16 English-German](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/models/wmt16.en-de.joined-dict.transformer.tar.bz2) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2)
Transformer <br> ([Edunov et al., 2018](https://arxiv.org/abs/1808.09381); WMT'18 winner) | [WMT'18 English-German](http://www.statmt.org/wmt18/translation-task.html) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/models/wmt18.en-de.ensemble.tar.bz2) | See NOTE in the archive

### Language models

Description | Dataset | Model | Test set(s)
---|---|---|---
Convolutional <br> ([Dauphin et al., 2017](https://arxiv.org/abs/1612.08083)) | [Google Billion Words](https://github.com/ciprian-chelba/1-billion-word-language-modeling-benchmark) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/models/gbw_fconv_lm.tar.bz2) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/data/gbw_test_lm.tar.bz2)
Convolutional <br> ([Dauphin et al., 2017](https://arxiv.org/abs/1612.08083)) | [WikiText-103](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/models/wiki103_fconv_lm.tar.bz2) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/data/wiki103_test_lm.tar.bz2)

### Stories

Description | Dataset | Model | Test set(s)
---|---|---|---
Stories with Convolutional Model <br> ([Fan et al., 2018](https://arxiv.org/abs/1805.04833)) | [WritingPrompts](https://arxiv.org/abs/1805.04833) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/models/stories_checkpoint.tar.bz2) | [download (.tar.bz2)](https://s3.amazonaws.com/fairseq-py/data/stories_test.tar.bz2)


### Usage

Generation with the binarized test sets can be run in batch mode as follows, e.g. for WMT 2014 English-French on a GTX-1080ti:
```
$ curl https://s3.amazonaws.com/fairseq-py/models/wmt14.v2.en-fr.fconv-py.tar.bz2 | tar xvjf - -C data-bin
$ curl https://s3.amazonaws.com/fairseq-py/data/wmt14.v2.en-fr.newstest2014.tar.bz2 | tar xvjf - -C data-bin
$ python generate.py data-bin/wmt14.en-fr.newstest2014  \
  --path data-bin/wmt14.en-fr.fconv-py/model.pt \
  --beam 5 --batch-size 128 --remove-bpe | tee /tmp/gen.out
...
| Translated 3003 sentences (96311 tokens) in 166.0s (580.04 tokens/s)
| Generate test with beam=5: BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)

# Scoring with score.py:
$ grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
$ grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
$ python score.py --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref
BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)
```

# Join the fairseq community

* Facebook page: https://www.facebook.com/groups/fairseq.users
* Google group: https://groups.google.com/forum/#!forum/fairseq-users

# License
fairseq(-py) is BSD-licensed.
The license applies to the pre-trained models as well.
We also provide an additional patent grant.

# Credits
This is a PyTorch version of
[fairseq](https://github.com/facebookresearch/fairseq), a sequence-to-sequence
learning toolkit from Facebook AI Research. The original authors of this
reimplementation are (in no particular order) Sergey Edunov, Myle Ott, and Sam
Gross.
