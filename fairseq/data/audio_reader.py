# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import h5py
import numpy as np
import torch

def get_reader(format):
    supported_types = {'h5': reader_h5,
                       'npz': reader_npz}

    return supported_types[format]

def reader_h5(path):
    with h5py.File(path, "r") as file:
        l = list(file.keys())
        l.sort(key=lambda x : int(x))
        for key in l:
            yield torch.from_numpy(file[str(key)].value)

def reader_npz(path):
    with open(path, 'rb') as f:
        shape = np.load(f)
        for i in range(int(shape[0])):
            yield torch.from_numpy(np.load(f))
