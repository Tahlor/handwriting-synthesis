import numpy as np
from utils import convert_gts_to_synth_format

def load_samples(path):
    samples = np.load(path, allow_pickle=True)
    return process_samples(samples)


def process_samples(samples):
    for sample in samples:
        convert_gts_to_synth_format(sample)

    return samples


if __name__ == "__main__":
    samples = load_samples('0.npy')
    print(samples)