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

ORIGINAL_MINE = ROOT / f"data/processed/original_mine"
ORIGINAL_THEIRS = ROOT / f"data/processed/original"
VERSION="4"
def combine(drop_bad=True, original=ORIGINAL_MINE):
    suffix = "mine" if "mine" in original.name else "theirs"
    var = f"offline_no_drop{VERSION}" if not drop_bad else f"offline_drop{VERSION}"
    new = ROOT / f"data/processed/{var}"
    combined = ROOT / f"data/processed/{var}/combined_{suffix}"
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
    return combined

def load(drop_bad=True, combined=None):
    from matplotlib import pyplot as plt
    var = f"offline_no_drop{VERSION}" if not drop_bad else f"offline_drop{VERSION}"
    if not combined:
        combined = ROOT / f"data/processed/{var}/combined"

    offsets = np.load(combined / "x.npy", allow_pickle=True)
    print("Original")
    # utils.plot_from_synth_format(offsets[0])
    # #input("New data")
    # utils.plot_from_synth_format(offsets[-1])

    for axis in 0,: #0,1:
        print(f"AXIS {axis}")
        other = offsets[0:1000][:,:,axis].flatten()
        print(other.shape)

        plt.hist(other, range=(-3,3),)
        plt.ylim(0,1e6)
        plt.title(f"X-coords Online Data {combined.stem}")
        plt.show()

        other = offsets[-1000:][:,:,axis].flatten()
        print(other.shape)
        plt.ylim(0,1e6)
        plt.hist(other, range=(-3,3))
        plt.title(f"X-coords Offline Data {combined.stem}")
        plt.show()


if __name__ == '__main__':
    for original in ORIGINAL_MINE, ORIGINAL_THEIRS:
    #for original in ORIGINAL_THEIRS,:
        combined=combine(drop_bad=True, original=original)
        load(drop_bad=True, combined=combined)

        combined=combine(drop_bad=False, original=original)
        load(drop_bad=False, combined=combined)

