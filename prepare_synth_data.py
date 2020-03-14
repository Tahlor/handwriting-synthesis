import time
import multiprocessing
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

def process(_sample):
    _stroke = convert_gts_to_synth_format(_sample['stroke'])
    _char = process_chars(_sample['text'])
    _id = "-".join(_sample['id'].split("-")[:2])
    return {'stroke': _stroke, 'char': _char, 'id': _id, 'text':_sample['text']}

def process_data(samples):
    global counter, strokes, chars, ids, text
    strokes, chars, ids, text = [], [], [], []
    counter = 0

    poolcount = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=poolcount)

    counter = 0

    def callback(stroke_dict):
        global counter
        counter += 1
        if len(stroke_dict["stroke"]) > drawing.MAX_STROKE_LEN:
            warnings.warn("stroke too long")
        elif stroke_dict["char"] is not None:
            strokes.append(stroke_dict["stroke"])
            chars.append(stroke_dict["char"])
            ids.append(stroke_dict["id"])
            text.append(stroke_dict['text'])

    # LOOP
    for sample in tqdm(samples):
        if True:
            pool.apply_async(func=process, args=(sample,), callback=callback)
        else:
            callback(process(sample))
    pool.close()

    # Track pool
    previous = 0
    with tqdm(total=len(samples)) as pbar:
        while previous < len(samples):
            time.sleep(1)
            new = counter
            pbar.update(new - previous)
            previous = new
    pool.join()

    return strokes, chars, ids, text

if __name__ == "__main__":
    data = load_data(DATA)
    strokes, chars, w_id, text= process_data(data)

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