# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import OrderedDict

import numpy as np

from . import FairseqDataset


class MultilingualDatasets(FairseqDataset):
    """Zip multiple FairseqDatasets together

    Args:
        datasets: a dictionary of FairseqDatasets
        eval_key: an optional key used at evaluation time that causes this
            instance to pass-through batches from `datasets[eval_key]`.
    """

    def __init__(self, datasets, lang_dict):
        super().__init__()
        assert isinstance(datasets, OrderedDict)
        self.datasets = list([dataset for k, dataset in datasets])
        self.langs = list([k for k, dataset in datasets])
        self.lens = [len(d) for d in self.datasets]
        self.lang_dict = lang_dict

    def _map_index(self, index):
        for bin in range(len(self.datasets)):
            if index < self.lens:
                break
        if bin == 0:
            low = 0
        else:
            low = self.lens[bin-1]
        return bin, index-low

    def __getitem__(self, index):
        key, index = self._map_index(index)
        return self.datasets[key][index]

    def __len__(self):
        return sum(self.lens)

    def collater(self, samples):
        #TODO
        """Merge a list of samples to form a mini-batch."""
        if len(samples) == 0:
            return None
        if self.eval_key is None:
            return OrderedDict([
                (key, dataset.collater([sample[key] for sample in samples]))
                for key, dataset in self.datasets.items()
            ])
        else:
            # at evaluation time it's useful to pass-through batches from a single key
            return self.datasets[self.eval_key].collater(samples)

    def get_dummy_batch(self, max_tokens, max_positions):
        batch = self.datasets[0].get_dummy_batch(max_tokens, max_positions[0])
        keys = self.lang_dict.dummy_sentence(batch.size(0))
        return batch, keys

    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        # TODO make it configurable whether to use max() or sum() here
        return max(
            dataset.num_tokens(self._map_index(key, index))
            for key, dataset in self.datasets.items()
        )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return {
            key: dataset.size(self._map_index(key, index))
            for key, dataset in self.datasets.items()
        }

    def ordered_indices(self):
        """Ordered indices for batching."""
        return np.arange(len(self))

    def valid_size(self, index, max_positions):
        """Check if an example's size is valid according to max_positions."""
        return all(
            dataset.valid_size(self._map_index(key, index), max_positions[key])
            for key, dataset in self.datasets.items()
        )

    def prefetch(self, indices):
        for _, dataset in self.datasets.items():
            dataset.prefetch(indices)

    @property
    def supports_prefetch(self):
        return all(
            [hasattr(dataset, 'supports_prefetch') and dataset.supports_prefetch
            for _, dataset in self.datasets.items()]
        )