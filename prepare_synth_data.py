from utils import *
import numpy as np
import drawing
from drawing import MAX_CHAR_LEN, MAX_STROKE_LEN
import utils

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

"""
Takes strokes recovered from offline data and prepares them to be used as training data

"""

DATA = get_folder("archidata/all_data_v2.npy")
#DATA = "archidata/0.npy"

def load_data(path):
    samples = np.load(path, allow_pickle=True)
    return samples

def process_stroke(stroke):
    new_stroke = stroke.copy()
    sps = new_stroke[:, -1]


    new_stroke[:, -1] = np.round(new_stroke[:, -1])
    new_stroke = sos_to_eos(new_stroke)


    coords = drawing.align(new_stroke)
    coords = drawing.denoise(coords)
    offsets = drawing.coords_to_offsets(coords)
    offsets = offsets[:drawing.MAX_STROKE_LEN]
    offsets = drawing.normalize(offsets)

    # Make sure we can draw it
    if False:
        test = offsets.copy()
        test[:,:2] = np.cumsum(test[:,:2], axis=0)

        test[:,1] -= np.min(test[:,1])
        test[:,:2] /= np.max(test[:,1])
        test = eos_to_sos(test)

        img = utils.draw_from_gt(test, use_stroke_number=False, show=True)

        from matplotlib import pyplot as plt
        plt.imshow(img)
        plt.show()

    return offsets

def process_chars(chars):
    chars = chars.strip()
    if any(c not in drawing.alphabet for c in chars):
        return None
    chars = drawing.encode_ascii(chars)[:MAX_CHAR_LEN]
    return chars

def process_data(samples):
    strokes, chars = [], []
    for sample in samples:
        stroke = process_stroke(sample['stroke'][:, 0:-1])
        char = process_chars(sample['text'])
        if char is not None:
            strokes.append(stroke)
            chars.append(char)
        else:
            continue
    return strokes, chars


if __name__ == "__main__":
    data = load_data(DATA)
    strokes, chars = process_data(data)

    x = np.zeros([len(strokes), drawing.MAX_STROKE_LEN, 3], dtype=np.float32)
    x_len = np.zeros([len(strokes)], dtype=np.int16)
    c = np.zeros([len(strokes), drawing.MAX_CHAR_LEN], dtype=np.int8)
    c_len = np.zeros([len(strokes)], dtype=np.int8)

    for i, (stroke, char) in enumerate(zip(strokes, chars)):
        x[i, :len(stroke), :] = stroke
        x_len[i] = len(stroke)

        c[i, :len(char)] = char
        c_len[i] = len(char)

    root = get_folder()
    np.save(root+'data/processed/x.npy', x)
    np.save(root+'data/processed/x_len.npy', x_len)
    np.save(root+'data/processed/c.npy', c)
    np.save(root+'data/processed/c_len.npy', c_len)
