# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from fairseq import utils
from fairseq.modules import MultiheadAttention

from . import (
    FairseqEncoder, FairseqIncrementalDecoder, FairseqModel,
    register_model, register_model_architecture,
)


@register_model('ast_seq2seq')
class ASTS2SModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-layers', type=int, metavar='EXPR',
                            help='encoder lstm layers')
        parser.add_argument('--encoder-convolutions', type=str, metavar='EXPR',
                            help='encoder layers [(dim, kernel_size), ...]')
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
                            help='How to comput the encoder hidden state to be passed to the decoder')
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
        parser.add_argument('--conv-1d', action='store_true', default=False,
                            help='If true, computes conv1D of the input signal, else conv2D.')

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

        decoder_embed_dict = None
        if args.decoder_embed_path:
            decoder_embed_dict = utils.parse_embedding(args.decoder_embed_path)
            utils.print_embed_overlap(decoder_embed_dict, task.target_dictionary)

        # replaced source_dict with target_dict
        encoder = ASTS2SEncoder(
            linear_dim=args.encoder_embed_dim,
            convolutions=eval(args.encoder_convolutions),
            layers=args.encoder_layers,
            dropout=args.dropout,
            max_positions=args.max_source_positions,
            normalization_constant=args.normalization_constant,
            last_state=args.encoder_state,
            weight_norm=args.weight_norm,
            learn_initial=args.learn_initial_state,
            conv_1d=args.conv_1d,
            audio_features=task.audio_features,
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
            max_positions=args.max_target_positions,
            scale_norm=args.scale_norm,
            scale=args.scale,
        )
        return ASTS2SModel(encoder, decoder)


class ASTS2SEncoder(FairseqEncoder):
    """Convolutional encoder"""

    def __init__(
        self, linear_dim=128, max_positions=4096,
        convolutions=((512, 3),) * 20, dropout=0.1, normalization_constant=0.5,
        left_pad=True, layers=3, stride=2, last_state='last', weight_norm=False,
        learn_initial=False, conv_1d=False, audio_features=0,):
        super().__init__({})
        self.dropout = dropout
        self.normalization_constant = normalization_constant
        self.left_pad = left_pad
        self.num_attention_layers = None

        self.max_pos = max_positions

        convolutions = extend_conv_spec(convolutions)
        self.linear_dim = linear_dim

        assert audio_features != 0, "Cannot extract number of audio features"

        self.audio_features = audio_features

        self.fc1 = Linear(self.audio_features, 2*linear_dim, dropout=dropout, weight_norm=weight_norm)
        self.fc2 = Linear(2*linear_dim, linear_dim, dropout=dropout, weight_norm=weight_norm)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.convolutions = nn.ModuleList()

        conv = Conv1D if conv_1d else Conv2D
        in_channels = linear_dim if conv_1d else 1
        for i, (out_channels, kernel_size, kernel_width) in enumerate(convolutions):
            if kernel_size % 2 == 1:
                padding = kernel_size // 2
            else:
                padding = 0
            self.convolutions.append(
                conv(in_channels, out_channels, kernel_size,
                        dropout=dropout, padding=padding, stride=stride,
                       weight_norm=weight_norm)
            )
            in_channels = out_channels

        self.recurrent_layers = layers
        self.learn_initial = learn_initial
        reduction = stride ** len(convolutions)
        if not conv_1d:
            lstm_dim = linear_dim // reduction * in_channels
        else:
            lstm_dim = out_channels
        self.recurrent = LSTM(lstm_dim, lstm_dim // 2, num_layers=layers,
                              dropout=dropout, bidirectional=True)
        self.last_state = last_state
        self.conv1d = conv_1d

    def forward(self, src_tokens, src_lengths):
        # src_features: B x T x C
        x = self.tanh(self.fc1(F.dropout(src_tokens, p=self.dropout, training=self.training)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.tanh(self.fc2(x))

        if not self.conv1d:
            # B x T x C -> B x 1 x T x C
            x = x.unsqueeze(1)
        else:
            # B x T x C -> B x C x T
            x = x.transpose(1, 2)
        # temporal convolutions
        for conv in self.convolutions:
            x = F.dropout(x, p=self.dropout, training=self.training)
            if conv.kernel_size[0] % 2 == 1:
                # padding is implicit in the conv
                x = conv(x)
            else:
                padding_l = (conv.kernel_size[0] - 1) // 2
                padding_r = conv.kernel_size[0] // 2
                x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
                x = conv(x)

            src_lengths = torch.ceil(src_lengths.float() / 2).long()
            #x = self.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        if not self.conv1d:
            # B x Cout x T x F -> T x B x C
            bsz, out_channels, time, feats = x.size()
            x = x.transpose(1, 2).contiguous().view(bsz, time, -1)\
                .contiguous().transpose(0, 1)
        else:
            # B x C x T -> T x B x C
            x = x.transpose(1, 2).contiguous().transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        if not hasattr(self, 'lstm_state') or self.lstm_state[0].size(1) != x.size(1):
            self.lstm_state = tuple((x.new_zeros(self.recurrent_layers*2, x.size(1), x.size(2)//2),
                                    x.new_zeros(self.recurrent_layers * 2, x.size(1), x.size(2) // 2)))
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
                encoder_out['encoder_out'][1].index_select(0, new_order),
                encoder_out['encoder_out'][2].index_select(0, new_order),
            )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return min(self.max_pos, 70000)

    def reshape_bidirectional_encoder_state(self, state):
        layersx2, bsz, hsz = state.size()
        return state.view(layersx2//2, 2, bsz, hsz) \
            .transpose(1, 2).contiguous().view(layersx2//2, bsz, -1).squeeze(0)


class DotAttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, output_embed_dim, bmm=None, function='softmax'):
        super().__init__()

        self.activ = lambda x: F.softmax(x, dim=1) if function == 'softmax' else F.sigmoid
        #self.output_proj = Linear(input_embed_dim + output_embed_dim, output_embed_dim, bias=False)
        self.bmm = bmm if bmm is not None else torch.bmm

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim

        # x: bsz x output_embed_dim
        x = input

        # compute attention
        attn_scores = self.bmm(x.unsqueeze(1), source_hids.transpose(0, 1) \
                .transpose(1, 2)).squeeze(1)

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        sz = attn_scores.size()
        attn_scores = self.activ(attn_scores)
        attn_scores = attn_scores.unsqueeze(1)

        # sum weighted sources
        x = self.bmm(attn_scores, source_hids.transpose(0, 1)).squeeze(1)
        return x, attn_scores.squeeze(1).transpose(0, 1)


class GeneralAttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, output_embed_dim,
                 bmm=None, weight_norm=True, function='softmax', scale=True,
                 scale_norm=True, epsilon=1e-6,):
        super().__init__()
        self.proj = Linear(output_embed_dim, output_embed_dim, weight_norm=weight_norm)
        self.activ = (lambda x: F.softmax(x, dim=1)) if function == 'softmax' else torch.sigmoid
        if function == 'sigmoid' or function == 'fix-sigmoid':
            self.max_energy = 1

        self.bmm = bmm if bmm is not None else torch.bmm
        self.scale_norm = scale_norm
        self.scale = scale
        self.eps = epsilon

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim

        # x: bsz x output_embed_dim
        x = self.proj(input)

        # compute attention  :
        attn_scores = self.bmm(x.unsqueeze(1), source_hids.transpose(0, 1) \
                                .transpose(1, 2)).squeeze(1)

        # don't attend over padding
        if encoder_padding_mask is not None:
            energies = attn_scores.masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back
        else:
            energies = attn_scores

        scores = self.activ(energies)
        attn_scores = scores.unsqueeze(1)
        attn_weights = attn_scores.transpose(0, 2)

        # sum weighted sources
        x = self.bmm(attn_scores, source_hids.transpose(0, 1)).squeeze(1)
        return x, attn_weights.transpose(0, 1)


class MLPAttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, output_embed_dim,
                 bmm=None, weight_norm=True, function='softmax', scale=True,
                 scale_norm=True, epsilon=1e-6, dropout=0.0, ):
        super().__init__()
        self.proj1 = Linear(output_embed_dim, output_embed_dim, weight_norm=weight_norm)
        self.projf = Linear(output_embed_dim, 1, bias=False, weight_norm=weight_norm)

        self.bmm = bmm if bmm is not None else torch.bmm
        self.eps = epsilon
        self.dropout = dropout

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim

        x = self.proj1(input.unsqueeze(0)) + source_hids
        attn_scores = self.projf(torch.tanh(x)).squeeze(2).t()

        # don't attend over padding
        if encoder_padding_mask is not None:
            energies = attn_scores.masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back
        else:
            energies = attn_scores

        scores = F.softmax(energies, dim=1)
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        attn_scores = scores.unsqueeze(1)
        attn_weights = attn_scores.transpose(0, 2)

        # sum weighted sources
        x = self.bmm(attn_scores, source_hids.transpose(0, 1)).squeeze(1)
        return x, attn_weights.transpose(0, 1)


class CLSTMDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""

    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None, attention_layer=DotAttentionLayer,
        initial_state='same', weight_norm=False, attention_function='softmax',
        num_heads=16, scale_norm=True, scale=True, max_positions=4000,
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.need_attn = True
        self.max_pos = max_positions

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        assert encoder_output_units == hidden_size, \
            'encoder_output_units ({}) != hidden_size ({})'.format(encoder_output_units, hidden_size)
        # TODO another Linear layer if not equal

        self.num_layers = num_layers
        self.layers = nn.ModuleList([
                                        LSTMCell(
                                            input_size=embed_dim if layer == 0 else hidden_size,
                                            hidden_size=hidden_size
                                        )
                                        for layer in range(num_layers)
                                        ])
        self.multi_head = attention_layer == MultiheadAttention
        if self.multi_head:
            self.attention = attention_layer(encoder_output_units, num_heads,
                                             dropout=dropout_out)
        else:
            self.attention = attention_layer(encoder_output_units, hidden_size,
                                         weight_norm=weight_norm,
                                         function=attention_function, scale=scale,
                                         scale_norm=scale_norm)\
                         if attention else None
        if isinstance(attention, MLPAttentionLayer):
            self.ctx_proj = Linear(hidden_size, hidden_size, weight_norm=weight_norm)
        self.additional_fc = Linear(2*hidden_size + embed_dim, out_embed_dim, weight_norm=weight_norm)
        self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out, weight_norm=weight_norm)
        self.tanh = nn.Tanh()
        self.initial_state = initial_state

        if self.initial_state == 'linear':
            self.proj_hidden = Linear(hidden_size, hidden_size, dropout=dropout_out, weight_norm=weight_norm)
            self.proj_cell = Linear(hidden_size, hidden_size, dropout=dropout_out, weight_norm=weight_norm)

    def forward(self, prev_output_tokens, encoder_out_dict, incremental_state=None):
        if encoder_out_dict is not None:
            encoder_out = encoder_out_dict['encoder_out']
            encoder_padding_mask = encoder_out_dict['encoder_padding_mask']

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, _, _ = encoder_out[:3]
        srclen = encoder_outs.size(0)

        if bsz != encoder_outs.size(1):
            prev_output_tokens = prev_output_tokens.t()
            bsz, seqlen = seqlen, bsz

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        x_in = F.dropout(x, p=self.dropout_in, training=self.training)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells = cached_state
        else:
            _, encoder_hiddens, encoder_cells = encoder_out[:3]
            if self.initial_state == 'same':
                prev_hiddens = encoder_hiddens
                prev_cells = encoder_cells
            elif self.initial_state == 'linear':
                prev_hiddens = self.tanh(self.proj_hidden(encoder_hiddens))
                prev_cells = self.tanh(self.proj_cell(encoder_cells))
            else:
                raise NotImplementedError()

        attn_scores = x.data.new(srclen, seqlen, bsz).zero_()
        outs = []
        ctxs = []

        if hasattr(self, 'ctx_proj'):
            encoder_ctx = self.ctx_proj(encoder_outs)
        else:
            encoder_ctx = encoder_outs
        for j in range(seqlen):
            input = x_in[j]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens, prev_cells))

                # apply attention using the last layer's hidden state
                if self.attention is not None and i == 0:
                    # attention output becomes the input to the next layer
                    attn_input = F.dropout(hidden, p=self.dropout_out, training=self.training)
                    ctx, attn_scores[:, j, :] = self.attention(attn_input, encoder_ctx, encoder_padding_mask)
                    ctxs.append(ctx)
                    input = F.dropout(ctx, p=self.dropout_out, training=self.training)
                else:
                    out = hidden
                # save state for next time step
                prev_hiddens = hidden
                prev_cells = cell

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state', (prev_hiddens, prev_cells))

        # collect outputs across time steps
        outs = torch.stack(outs)
        ctxs = torch.stack(ctxs)

        out = torch.cat([outs, ctxs, x], dim=2)
        out = F.dropout(out, p=self.dropout_out, training=self.training)

        # T x B x C -> B x T x C
        out = out.transpose(1, 0)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        # project back to size of vocabulary
        out = self.tanh(self.additional_fc(out))
        out = F.dropout(out, p=self.dropout_out, training=self.training)
        out = self.fc_out(out)

        return out, attn_scores

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def _embed_tokens(self, tokens, incremental_state):
        if incremental_state is not None:
            # keep only the last token for incremental forward pass
            tokens = tokens[:, -1:]
        return self.embed_tokens(tokens)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return min(self.max_pos, 70000)


def extend_conv_spec(convolutions):
    """
    Extends convolutional spec that is a list of tuples of 2 or 3 parameters
    (kernel size, dim size and optionally how many layers behind to look for residual)
    to default the residual propagation param if it is not specified
    """
    extended = []
    for spec in convolutions:
        if len(spec) == 3:
            extended.append(spec)
        elif len(spec) == 2:
            extended.append(spec + (1,))
        else:
            raise Exception('invalid number of parameters in convolution spec ' + str(spec) + '. expected 2 or 3')
    return tuple(extended)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, dropout=0, bias=True, weight_norm=False):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    nn.init.normal_(m.weight, mean=0, std=math.sqrt((1 - dropout) / in_features))
    if bias:
        nn.init.constant_(m.bias, 0)
    if weight_norm:
        return nn.utils.weight_norm(m)
    else:
        return m


def Conv1D(in_channels, out_channels, kernel_size, dropout=0, weight_norm=False, **kwargs):
    """Weight-normalized Conv1d layer"""
    m = nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    if weight_norm:
        return nn.utils.weight_norm(m, dim=2)
    else:
        return m

def Conv2D(in_channels, out_channels, kernel_size, dropout=0, weight_norm=False, **kwargs):
    """Weight-normalized Conv2d layer"""
    m = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
    nn.init.normal_(m.weight, mean=0, std=0.1)
    nn.init.constant_(m.bias, 0)
    if weight_norm:
        return nn.utils.weight_norm(m, dim=2)
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


@register_model_architecture('ast_seq2seq', 'ast_seq2seq')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_convolutions = getattr(args, 'encoder_convolutions', '[(16, 3, 3)] * 2')
    args.encoder_layers = 3
    args.encoder_state = getattr(args, 'encoder_state', 'last')
    args.learn_initial_state = getattr(args, 'learn_initial_state', False)
    args.normalization_constant = getattr(args, 'normalization_constant', 0.5)
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

@register_model_architecture('ast_seq2seq', 'ast_seq2seq_red8')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_convolutions = getattr(args, 'encoder_convolutions', '[(32, 3, 3)] * 3')
    args.encoder_layers = 3
    args.encoder_state = getattr(args, 'encoder_state', 'avg')
    args.learn_initial_state = getattr(args, 'learn_initial_state', False)
    args.normalization_constant = getattr(args, 'normalization_constant', 0.5)
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

@register_model_architecture('ast_seq2seq', 'ast_seq2seq_red16')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_convolutions = getattr(args, 'encoder_convolutions', '[(64, 3, 3)] * 4')
    args.encoder_layers = 3
    args.encoder_state = getattr(args, 'encoder_state', 'avg')
    args.learn_initial_state = getattr(args, 'learn_initial_state', False)
    args.normalization_constant = getattr(args, 'normalization_constant', 0.5)
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
