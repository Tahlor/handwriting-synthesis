from tools import distortions
import copy
import utils

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataFrame(object):

    """Minimal pd.DataFrame analog for handling n-dimensional numpy matrices with additional
    support for shuffling, batching, and train/test splitting.

    Args:
        columns: List of names corresponding to the matrices in data.
        data: List of n-dimensional data matrices ordered in correspondence with columns.
            All matrices must have the same leading dimension.  Data can also be fed a list of
            instances of np.memmap, in which case RAM usage can be limited to the size of a
            single batch.
    """

    def __init__(self, columns, data, warp=True):
        assert len(columns) == len(data), 'columns length does not match data length'

        lengths = [mat.shape[0] for mat in data]
        assert len(set(lengths)) == 1, 'all matrices in data must have same first dimension'

        self.length = lengths[0]
        self.columns = columns
        self.data = data
        self.dict = dict(zip(self.columns, self.data))
        self.idx = np.arange(self.length)
        self.warp = warp

    def to_numpy(self):
        pd.DataFrame.to_numpy(self)

    def shapes(self):
        return pd.Series(dict(zip(self.columns, [mat.shape for mat in self.data])))

    def dtypes(self):
        return pd.Series(dict(zip(self.columns, [mat.dtype for mat in self.data])))

    def shuffle(self):
        np.random.shuffle(self.idx)

    def train_test_split(self, train_size, random_state=np.random.randint(1000), stratify=None):
        train_idx, test_idx = train_test_split(
            self.idx,
            train_size=train_size,
            random_state=random_state,
            stratify=stratify
        )
        train_df = DataFrame(copy.copy(self.columns), [mat[train_idx] for mat in self.data])
        test_df = DataFrame(copy.copy(self.columns), [mat[test_idx] for mat in self.data])
        return train_df, test_df

    def batch_generator(self, batch_size, shuffle=True, num_epochs=10000, allow_smaller_final_batch=False):
        epoch_num = 0
        while epoch_num < num_epochs:
            if shuffle:
                self.shuffle()

            for i in range(0, self.length + 1, batch_size): # loop through all items using batch_size step
                batch_idx = self.idx[i: i + batch_size]
                if not allow_smaller_final_batch and len(batch_idx) != batch_size:
                    break

                if self.warp:
                    data = [mat[batch_idx].copy() for mat in self.data]

                    coords_batch = data[0] # list, ['x', 'x_len', 'c', 'c_len', 'text']
                    for ii, item in enumerate(coords_batch):
                        #utils.plot_from_synth_format(item)
                        gt, xmin, ymin, factor = utils.convert_synth_offsets_to_gt(item, return_all=True)
                        gt[:,0:2] = distortions.warp_points(gt*61)/61

                        gt[:, 0:2] *= factor
                        gt[:, 1] += ymin  # min_y = 0
                        gt[:, 0] += xmin  # min_x = 0
                        
                        coords_batch[ii] = utils.convert_gts_to_synth_format(gt, adjustments=True)
                        utils.plot_from_synth_format(coords_batch[ii])
                        continue
                yield DataFrame(
                    columns=copy.copy(self.columns),
                    data=data
                )

            epoch_num += 1

    def iterrows(self):
        for i in self.idx:
            yield self[i]

    def mask(self, mask):
        return DataFrame(copy.copy(self.columns), [mat[mask] for mat in self.data])

    def concat(self, other_df):
        mats = []
        for column in self.columns:
            mats.append(np.concatenate([self[column], other_df[column]], axis=0))
        return DataFrame(copy.copy(self.columns), mats)

    def items(self):
        return self.dict.items()

    def __iter__(self):
        return self.dict.items().__iter__()

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.dict[key]

        elif isinstance(key, int):
            return pd.Series(dict(zip(self.columns, [mat[self.idx[key]] for mat in self.data])))

    def __setitem__(self, key, value):
        assert value.shape[0] == len(self), 'matrix first dimension does not match'
        if key not in self.columns:
            self.columns.append(key)
            self.data.append(value)
        self.dict[key] = value

def test():
    gt =          [[0, 1, 1],
                   [2, 5, 0],
                   [8, 9, 0],
                   [20, 21, 1],
                   [12, 13, 1],
                   [16, 17, 0],
                   [32, 33, 1],
                   [28, 29, 0],
                   [24, 15, 0]]
    gt = np.asarray(gt).astype(np.float64)
    # gt[:, 1] -= np.min(gt[:, 1]) # min_y = 0
    # gt[:, 0] -= np.min(gt[:, 0]) # min_x = 0
    #gt[:, :2] = gt[:, :2]/np.max(gt[:, 1])

    offsets = utils.convert_gts_to_synth_format(gt, adjustments=True)
    gt2 = utils.convert_synth_offsets_to_gt(offsets)
    offsets2 = utils.convert_gts_to_synth_format(gt2, adjustments=False)
    gt3 = utils.convert_synth_offsets_to_gt(offsets2)
    np.set_printoptions(suppress=True)
    print(gt)
    print(gt2)
    print(gt3)
    print(offsets)
    print(offsets2)

if __name__=='__main__':

    # Draw a thing
    ROOT = utils.get_project_root()

    original = ROOT / f"data/processed/original_mine"
    file = "x.npy"
    np.load(original / file, allow_pickle=True)
