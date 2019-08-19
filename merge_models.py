from fairseq import options
from strip_modules import (
    save_state, load_model_state,
)
from collections import OrderedDict

def main(args):
    encoder_state = load_model_state(args.encoder)
    decoder_state = load_model_state(args.decoder)
    model_state = encoder_state
    for k, v in decoder_state['model'].items():
        if k.startswith('decoder'):
            model_state['model'][k] = v
    model_state['modified'] = True
    save_state(model_state, args.new_model_path)

if __name__ == '__main__':
    parser = options.get_parser('Merge models')
    options.add_merging_args(parser)

    args = parser.parse_args()
    main(args)
