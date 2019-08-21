import math
import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models.r_transformer import TransformerEncoder
from fairseq.models.ast_seq2seq import (
        CLSTMDecoder, DotAttentionLayer, GeneralAttentionLayer,
        MLPAttentionLayer,)

from fairseq.modules import (
    MultiheadAttention,
)

from . import (
    FairseqModel, register_model, register_model_architecture,
)


@register_model('transf2ber')
class Transf2BerModel(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--relu-dropout', type=float, metavar='D',
                            help='dropout probability after ReLU in FFN')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. Must be used with adaptive_loss criterion')
        parser.add_argument('--encoder-convolutions', type=str, metavar='EXPR',
                            help='encoder layers [(dim, kernel_size), ...]')

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
                            help='How to comput the ncoder hidden state to be passed to the decoder')
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
        parser.add_argument('--hidden-size', type=int, help='RNN units in the decoder')
        parser.add_argument('--encoder-output-units', type=int, help='dimension of encoder output embeddings')


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        attention_layer = {'dot': DotAttentionLayer,
                           'general': GeneralAttentionLayer,
                           'multi-head': MultiheadAttention,
                           'mlp': MLPAttentionLayer,
                          }

        if args.decoder_embed_path:
            decoder_embed_dict = utils.parse_embedding(args.decoder_embed_path)
            utils.print_embed_overlap(decoder_embed_dict, task.target_dictionary)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        def build_audio_embedding(embed_dim, dropout):
            m = nn.Linear(task.audio_features, embed_dim)
            nn.init.normal_(m.weight, mean=0, std=math.sqrt((1 - dropout) / task.audio_features))
            nn.init.constant_(m.bias, 0)
            return m


        encoder_embed_tokens = build_audio_embedding(
                2*args.encoder_embed_dim, args.dropout
        )

        encoder = ProxyEncoder(args,
                tgt_dict,
                encoder_embed_tokens,
                audio_features=task.audio_features,
                )
        decoder = CLSTMDecoder(
                dictionary=task.target_dictionary,
                embed_dim=args.decoder_embed_dim,
                out_embed_dim=args.decoder_out_embed_dim,
                attention=eval(args.decoder_attention),
                hidden_size=args.hidden_size,
                dropout_in=args.dropout,
                dropout_out=args.dropout,
                num_layers=args.decoder_layers,
                attention_layer=attention_layer[args.attention_type],
                initial_state=args.decoder_initial_state,
                weight_norm=args.weight_norm,
                attention_function=args.attention_function,
                scale_norm=args.scale_norm,
                )
        return Transf2BerModel(encoder, decoder)


class ProxyEncoder(TransformerEncoder):

    def __init__(self, args, src_dict, embed_tokens, audio_features):
        super().__init__(args=args, dictionary=src_dict, embed_tokens=embed_tokens, audio_features=audio_features)
        self.final_proj_out = Linear(256, 512)
        self.final_proj_h = Linear(256, 512)
        self.final_proj_c = Linear(256, 512)

    def forward(self, src_features, src_lengths):
        encoder_out = super().forward(src_features, src_lengths)
        encoder_out['encoder_out'] = (torch.tanh(self.final_proj_out(encoder_out['encoder_out'])),
                torch.tanh(self.final_proj_h(encoder_out['encoder_out'][-1])),
                torch.tanh(self.final_proj_c(encoder_out['encoder_out'].sum(dim=0) / encoder_out['encoder_out'].size(0))))
        return encoder_out

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


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m

@register_model_architecture('transf2ber', 'transf2ber')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.normalization_constant = getattr(args, 'normalization_constant', 0.5)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.relu_dropout = getattr(args, 'relu_dropout', 0.1)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)

    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 128)
    args.encoder_convolutions = getattr(args, 'encoder_convolutions', '[(8, 3, 3)] * 2')
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 768)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.no_token_positional_embeddings = False

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
    args.hidden_size = getattr(args, 'hidden_size', 512)
