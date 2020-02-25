import numpy as np


def load_samples(path):
    samples = np.load(path, allow_pickle=True)
    return process_samples(samples)


def process_samples(samples):
    for sample in samples:
        sample['stroke'] = sample['stroke'][:, 0:-1]
        sps = sample['stroke'][:, -1]
        sample['stroke'][:, -1] = np.where(np.diff(np.insert(sps, 0, 0)) > 0.5, 1, 0)
    return samples


if __name__ == "__main__":
    samples = load_samples('0.npy')
    print(samples)