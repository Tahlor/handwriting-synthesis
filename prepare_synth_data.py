from tqdm import tqdm
from utils import *
import numpy as np
import drawing
from drawing import MAX_CHAR_LEN, MAX_STROKE_LEN
import utils

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

"""
Takes strokes recovered from offline data and prepares them to be used as training data

"""

DATA = get_folder("archidata/all_data_v3.npy")
#DATA = get_folder("/media/data/GitHub/simple_hwr/RESULTS/OFFLINE_PREDS/normal_preload_model/imgs/current/eval/data/55.npy")
#DATA = "archidata/0.npy"

def load_data(path):
    samples = np.load(path, allow_pickle=True)
    return samples

def process_stroke(stroke):
    return utils.convert_gts_to_synth_format(stroke)

def process_chars(chars):
    chars = chars.strip()
    if any(c not in drawing.alphabet for c in chars) or len(chars) > MAX_CHAR_LEN:
        return None
    chars = drawing.encode_ascii(chars)[:MAX_CHAR_LEN]
    return chars

def process_data(samples):
    strokes, chars = [], []
    for sample in tqdm(samples):
        stroke = convert_gts_to_synth_format(sample['stroke'])
        char = process_chars(sample['text'])
        id = "-".join(sample['id'].split("-")[:2])
        if char is not None:
            strokes.append(stroke)
            chars.append(char)
        else:
            continue
    return strokes, chars, id

if __name__ == "__main__":
    data = load_data(DATA)
    strokes, chars, w_id = process_data(data)
    text = [d["text"] for d in data]
    x = np.zeros([len(strokes), drawing.MAX_STROKE_LEN, 3], dtype=np.float32)
    x_len = np.zeros([len(strokes)], dtype=np.int16)
    c = np.zeros([len(strokes), drawing.MAX_CHAR_LEN], dtype=np.int8)
    c_len = np.zeros([len(strokes)], dtype=np.int8)

    for i, (stroke, char) in tqdm(enumerate(zip(strokes, chars))):
        x[i, :len(stroke), :] = stroke
        x_len[i] = len(stroke)

        c[i, :len(char)] = char
        c_len[i] = len(char)

    root = get_folder()
    np.save(root+'data/processed/x.npy', x)
    np.save(root+'data/processed/text.npy', text)
    np.save(root+'data/processed/x_len.npy', x_len)
    np.save(root+'data/processed/c.npy', c)
    np.save(root+'data/processed/c_len.npy', c_len)
    np.save(root + 'data/processed/w_id.npy', w_id)
    np.save(root+'data/processed/text.npy', text)
    print("Valid strokes", len(x))