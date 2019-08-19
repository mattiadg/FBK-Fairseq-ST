# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from fairseq import utils
from fairseq.modules import (
    MultiheadAttention
)

from . import (
    FairseqEncoder,FairseqModel,
    register_model, register_model_architecture,
)

from .ast_seq2seq import (
    CLSTMDecoder, GeneralAttentionLayer, DotAttentionLayer, MLPAttentionLayer
)


@register_model('bahdanau')
class LSTMCDecoder(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-recurrent-dim', type=int, metavar='N',
                            help='encoder LSTM dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-layers', type=int, metavar='EXPR',
                            help='encoder lstm layers')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-layers', type=str, metavar='EXPR',
                            help='decoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='EXPR',
                            help='decoder attention [True, ...]')
        parser.add_argument('--normalization-constant', type=float, metavar='D',
                            help='multiplies the result of the residual block by sqrt(value)')
        parser.add_argument('--share-input-output-embed', action='store_true',
                            help='share input and output embeddings (requires'
                                 ' --decoder-out-embed-dim and --decoder-embed-dim'
                                 ' to be equal)')
        parser.add_argument('--attention-type', metavar='T',
                            help='The function to compute attention energy')
        parser.add_argument('--encoder-state', metavar='STATE',
                            help='How to compute the encoder hidden state to be passed to the decoder')
        parser.add_argument('--decoder-initial-state', metavar='STATE',
                            help='Initializer of thed decoder state')
        parser.add_argument('--no-weight-norm', action='store_true',
                            help='Deactivate layer normalization')
        parser.add_argument('--learn-initial-state', action='store_true',
                            help='If True, the encoder learns the LSTM initial state')
        parser.add_argument('--attention-function', help='Softmax or sigmoid')
        parser.add_argument('--scale-norm', action='store_true', default=False,
                            help='For sigmoidal attention activation, scales the output by the norm')
        parser.add_argument('--no-scale', action='store_true', help='Do not scale sigmoidal activation')


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        attention_layer = {'dot': DotAttentionLayer,
                           'general': GeneralAttentionLayer,
                           'multi-head': MultiheadAttention,
                           'mlp': MLPAttentionLayer,
                          }

        # replaced source_dict with target_dict
        encoder = ASTS2SEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_recurrent_dim,
            pretrained_embed=None,
            layers=args.encoder_layers,
            dropout=args.dropout,
            normalization_constant=0,
            last_state=args.encoder_state,
            weight_norm=args.weight_norm,
            learn_initial=args.learn_initial_state,

        )
        decoder = CLSTMDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            out_embed_dim=args.decoder_out_embed_dim,
            attention=eval(args.decoder_attention),
            dropout_in=args.dropout,
            dropout_out=args.dropout,
            num_layers=args.decoder_layers,
            attention_layer=attention_layer[args.attention_type],
            initial_state=args.decoder_initial_state,
            weight_norm=args.weight_norm,
            attention_function=args.attention_function,
            scale_norm=args.scale_norm,
            scale=args.scale,
        )
        return LSTMCDecoder(encoder, decoder)


class ASTS2SEncoder(FairseqEncoder):
    """Convolutional encoder"""

    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, pretrained_embed=None,
        dropout=0.1, normalization_constant=0.5,
        left_pad=True, layers=3, last_state='last', weight_norm=False,
        learn_initial=False, padding_idx=0):
        super().__init__({})
        self.dropout = dropout
        self.normalization_constant = normalization_constant
        self.left_pad = left_pad
        self.num_attention_layers = None

        num_embeddings = len(dictionary)
        self.padding_idx = padding_idx
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.hidden_size = hidden_size

        self.recurrent_layers = layers
        self.learn_initial = learn_initial
        self.recurrent = LSTM(embed_dim, self.hidden_size // 2, num_layers= layers,
                              dropout=dropout, bidirectional=True)
        self.last_state = last_state


    def forward(self, src_tokens, src_lengths):
        src_tokens = src_tokens.t()
        if self.left_pad:
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                self.padding_idx,
                left_to_right=True,
            )
        # src_tokens: B x T
        x = F.dropout(self.embed_tokens(src_tokens), p=self.dropout, training=self.training)

        # pack embedded source tokens into a PackedSequence
        if not hasattr(self, 'lstm_state') or self.lstm_state[0].size(1) != x.size(1):
            self.lstm_state = tuple((x.new_zeros(self.recurrent_layers*2, x.size(1), x.size(2)),
                                    x.new_zeros(self.recurrent_layers * 2, x.size(1), x.size(2))))
            for state in self.lstm_state:
                nn.init.normal_(state, mean=0, std=0.1)
            self.lstm_state = tuple((Parameter(state, requires_grad=self.learn_initial) for state in self.lstm_state))

        x, s = self.recurrent(x, self.lstm_state)

        # unpack outputs and apply dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        encoder_padding_mask = self.create_mask(src_lengths)

        if encoder_padding_mask is not None:
            x = x.masked_fill_(
                encoder_padding_mask.transpose(0, 1).unsqueeze(-1),
                0.0
            ).type_as(x)

        if self.last_state == 'last':
            encoder_hiddens = self.reshape_bidirectional_encoder_state(s[0][-2:, ::])
            encoder_cells = self.reshape_bidirectional_encoder_state(s[1][-2:, ::])
        elif self.last_state == 'avg':
            encoder_hiddens = x.sum(dim=0) / x.size(0)
            encoder_cells = self.reshape_bidirectional_encoder_state(s[1][-2:, ::])
        else:
            raise NotImplementedError()

        return {
            'encoder_out': (x, encoder_hiddens, encoder_cells),
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def create_mask(self, lengths):
        max_len = max(lengths)
        mask = lengths.new_zeros(len(lengths), max_len).byte()
        for i, l in enumerate(lengths):
            mask[i, :max_len-l] = 1
        if not mask.any():
            mask = None
        return mask

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = (
                encoder_out['encoder_out'][0].index_select(1, new_order),
                encoder_out['encoder_out'][1].index_select(1, new_order),
                encoder_out['encoder_out'][2].index_select(1, new_order),
            )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return 100000
        #return self.embed_positions.max_positions() if self.embed_positions is not None else float('inf')

    def reshape_bidirectional_encoder_state(self, state):
        layersx2, bsz, hsz = state.size()
        return state.view(layersx2//2, 2, bsz, hsz) \
            .transpose(1, 2).contiguous().view(layersx2//2, bsz, -1).squeeze(0)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, dropout=0, bias=True, weight_norm=False):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    nn.init.normal_(m.weight, mean=0, std=math.sqrt((1 - dropout) / in_features))
    #nn.init.normal_(m.weight, mean=0, std=0.1)
    if bias:
        nn.init.constant_(m.bias, 0)
    if weight_norm:
        return nn.utils.weight_norm(m)
    else:
        return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param, mean=0, std=0.1)
        elif 'bias' in name:
            nn.init.constant_(param, 0)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param, mean=0, std=0.1)
        elif 'bias' in name:
            nn.init.constant_(param, 0)
            #param.data.uniform_(-0.1, 0.1)
    return m


@register_model_architecture('bahdanau', 'bahdanau')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_recurrent_dim = getattr(args, 'encoder_recurrent_dim', 512)
    args.encoder_layers = 1
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_state = getattr(args, 'encoder_state', 'avg')
    args.learn_initial_state = getattr(args, 'learn_initial_state', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_layers = 2
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    args.decoder_initial_state = getattr(args, 'decoder_initial_state', 'same')
    args.attention_type = getattr(args, 'attention_type', 'general')
    args.weight_norm = not getattr(args, 'no_weight_norm', False)
    args.attention_function = getattr(args, 'attention_function', 'softmax')
    args.scale_norm = getattr(args, 'scale_norm', False)
    args.scale = not getattr(args, 'no_scale', False)
