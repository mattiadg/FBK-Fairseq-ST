# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn as nn

from fairseq import utils
from fairseq.models.transformer import (
    TransformerDecoder,
)

from . import (
    FairseqModel, register_model, register_model_architecture,
)

from fairseq.models.ast_seq2seq import (
    ASTS2SEncoder
)



@register_model('ber2transf')
class Ber2Transf(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-dropout', type=float, metavar='D',
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
        parser.add_argument('--no-weight-norm', action='store_true',
                            help='Deactivate layer normalization')
        parser.add_argument('--learn-initial-state', action='store_true',
                            help='If True, the encoder learns the LSTM initial state')

        parser.add_argument('--decoder-dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. ')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 4000
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        tgt_dict = task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.decoder_embed_path:
            raise NotImplementedError("Pretrained embedding not available with Ber2Transf yet")
        else:
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = ProxyEncoder(
            linear_dim=args.encoder_embed_dim,
            convolutions=eval(args.encoder_convolutions),
            layers=args.encoder_layers,
            dropout=args.encoder_dropout,
            max_positions=args.max_source_positions,
            normalization_constant=args.normalization_constant,
            weight_norm=args.weight_norm,
            audio_features=task.audio_features,
        )
        args.dropout = args.decoder_dropout
        decoder = TransformerDecoder(
                args,
                tgt_dict,
                decoder_embed_tokens,
        )

        return Ber2Transf(encoder, decoder)


class ProxyEncoder(ASTS2SEncoder):
    """Convolutional encoder"""

    def __init__(
        self, linear_dim=128, max_positions=4096,
        convolutions=((512, 3),) * 20, dropout=0.1, normalization_constant=0.5,
        left_pad=True, layers=3, stride=2, weight_norm=False, audio_features=0,
    ):
        super().__init__(linear_dim=linear_dim, max_positions=max_positions,
                         convolutions=convolutions, dropout=dropout,
                         normalization_constant=normalization_constant,
                         left_pad=left_pad, layers=layers, stride=stride,
                         weight_norm=weight_norm, audio_features=audio_features,
                         )
        self.linear_out = Linear(512, 256)

    def forward(self, src_tokens, src_lengths):
        encoder_out = super().forward(src_tokens, src_lengths)
        if isinstance(encoder_out['encoder_out'], tuple):
            encoder_out['encoder_out'] = encoder_out['encoder_out'][0] # T x B x C
        encoder_out['encoder_out'] = self.linear_out(encoder_out['encoder_out'])

        return encoder_out

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out


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


def Linear(in_features, out_features, dropout=0, bias=True):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    nn.init.normal_(m.weight, mean=0, std=math.sqrt((1 - dropout) / in_features))
    if bias:
        nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m)


@register_model_architecture('ber2transf', 'ber2transf')
def base_architecture(args):
    args.encoder_dropout = getattr(args, 'encoder_dropout', 0.2)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_convolutions = getattr(args, 'encoder_convolutions', '[(16, 3, 3)] * 2')
    args.encoder_state = getattr(args, 'encoder_state', 'avg')
    args.encoder_layers = getattr(args, 'encoder_layers', 3)
    args.normalization_constant = getattr(args, 'normalization_constant', 0.5)
    args.weight_norm = not getattr(args, 'no_weight_norm', False)

    # with transf
    args.decoder_dropout = getattr(args, 'decoder_dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)

    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 768)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
