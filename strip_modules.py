import torch

from torch.serialization import default_restore_location
from fairseq.utils import _upgrade_state_dict, convert_state_dict_type, torch_persistent_save
from collections import OrderedDict

import os
import argparse

def load_model_state(filename):
    if not os.path.exists(filename):
        return None
    state = torch.load(filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    state = _upgrade_state_dict(state)

    return state

def _strip_params(state, strip_what='decoder'):
    new_state = state
    new_state['model'] = OrderedDict({key: value for key, value in state['model'].items()
                             if not key.startswith(strip_what)})

    return new_state

def save_state(state, filename):
    torch_persistent_save(state, filename)


def main(args):
    model_state = load_model_state(args.model_path)
    print("Loaded model {}".format(args.model_path))
    model_state = _strip_params(model_state, strip_what=args.strip_what)
    print("Stripped {}".format(args.strip_what))
    save_state(model_state, args.new_model_path)
    print("Saved to {}".format(args.new_model_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help="The path to the model to strip")
    parser.add_argument('--new-model-path', help="The name for the stripped model")
    parser.add_argument('--strip-what', choices=['encoder', 'decoder'],
                        help="Part of the network to strip away.")

    args = parser.parse_args()
    main(args)