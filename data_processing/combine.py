from __future__ import print_function
import time, multiprocessing
from pathlib import Path
import os
from xml.etree import ElementTree
from tqdm import tqdm
import numpy as np
import utils
import drawing

ROOT = utils.get_project_root()

original = ROOT / f"data/processed/original_mine"

def combine(drop_bad=True):
    var = "offline_no_drop" if not drop_bad else "offline_drop"
    new = ROOT / f"data/processed/{var}"
    combined = ROOT / f"data/processed/{var}/combined"
    combined.mkdir(exist_ok=True, parents=True)

    #for file in ['x.npy', "x_len.npy", 'c.npy', 'c_len.npy', 'text.npy']: #'w_id.npy',
    x = 0
    for file in ['x.npy', "x_len.npy", 'c.npy', 'c_len.npy', 'text.npy', 'w_id.npy']:
        comb = np.concatenate([np.load(original / file, allow_pickle=True), np.load(new / file, allow_pickle=True)], axis=0)
        np.save(combined / file, comb)

        if x:
            assert len(comb) == x
        else:
            x = len(comb)

def load(drop_bad=True):
    from matplotlib import pyplot as plt
    var = "offline_no_drop" if not drop_bad else "offline_drop"
    combined = ROOT / f"data/processed/{var}/combined"

    offsets = np.load(combined / "x.npy", allow_pickle=True)
    print("Original")
    utils.plot_from_synth_format(offsets[0])
    #input("New data")
    utils.plot_from_synth_format(offsets[-1])

    for axis in 0,1:
        print(f"AXIS {axis}")
        other = offsets[0:1000][:,:,axis].flatten()
        print(other.shape)

        plt.hist(other, range=(-3,3))
        plt.show()

        other = offsets[-1000:][:,:,axis].flatten()
        print(other.shape)
        plt.hist(other, range=(-3,3))
        plt.show()


if __name__ == '__main__':
    combine(drop_bad=True)
    load(drop_bad=True)

    combine(drop_bad=False)
    load(drop_bad=False)
