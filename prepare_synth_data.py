import time
import multiprocessing
from tqdm import tqdm
from utils import *
import numpy as np
import drawing
from drawing import MAX_CHAR_LEN, MAX_STROKE_LEN
import utils
import argparse
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

"""
Takes strokes recovered from offline data and prepares them to be used as training data

"""

#DATA = get_folder("archidata/all_data_v3.npy")
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
    return {'stroke': _stroke, 'char': _char, 'id': _id, 'text':_sample['text'], 'distance':_sample['distance']}

def process_data(samples, drop_bad=False, parallel=True):
    global counter, strokes, chars, ids, text
    strokes, chars, ids, text = [], [], [], []
    counter = 0

    poolcount = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=poolcount)

    counter = 0

    def callback(stroke_dict):
        global counter
        counter += 1
        if stroke_dict:
            if len(stroke_dict["stroke"]) > drawing.MAX_STROKE_LEN:
                warnings.warn("stroke too long")
            elif drop_bad and "distance" in stroke_dict and stroke_dict["distance"]>.01:
                warnings.warn("stroke error too high")
            elif np.any(np.linalg.norm(stroke_dict["stroke"][:, :2], axis=1) > 60):
                warnings.warn("Line too long")
            elif stroke_dict["char"] is not None:
                strokes.append(stroke_dict["stroke"])
                chars.append(stroke_dict["char"])
                ids.append(stroke_dict["id"])
                text.append(stroke_dict['text'])

    # LOOP
    for sample in tqdm(samples):
        if parallel:
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

def main(args):
    root = get_folder()
    variant = f"processed/offline_drop{args.version}" if args.drop_bad else f"processed/offline_no_drop{args.version}"
    (Path(root) / f"data/{variant}").mkdir(exist_ok=True, parents=True)

    data = load_data(args.data)
    strokes, chars, w_id, text= process_data(data, drop_bad=args.drop_bad)

    x = np.zeros([len(strokes), drawing.MAX_STROKE_LEN, 3], dtype=np.float32) # BATCH, WIDTH, (X,Y,EOS)
    x_len = np.zeros([len(strokes)], dtype=np.int16)
    c = np.zeros([len(strokes), drawing.MAX_CHAR_LEN], dtype=np.int8)
    c_len = np.zeros([len(strokes)], dtype=np.int8)

    for i, (stroke, char) in tqdm(enumerate(zip(strokes, chars))):
        x[i, :len(stroke), :] = stroke
        x_len[i] = len(stroke)

        c[i, :len(char)] = char
        c_len[i] = len(char)

    np.save(root+f'data/{variant}/x.npy', x)
    np.save(root+f'data/{variant}/text.npy', text)
    np.save(root+f'data/{variant}/x_len.npy', x_len)
    np.save(root+f'data/{variant}/c.npy', c)
    np.save(root+f'data/{variant}/c_len.npy', c_len)
    np.save(root+f'data/{variant}/w_id.npy', w_id)
    np.save(root+f'data/{variant}/text.npy', text)
    print("Valid strokes", len(x))

if __name__ == "__main__":
    # DATA = "archidata/adapted_dtw_v2.npy"
    # VERSION = 3
    source_version = {4:"RESUME_model/imgs/current/eval1/data/all_data.npy",
                      5: "RESUME_model/imgs/current/eval2/data/all_data.npy"
                      }

    VERSION = 4
    DATA = Path("/media/data/GitHub/simple_hwr/RESULTS/OFFLINE_PREDS") / source_version[VERSION]

    parser = argparse.ArgumentParser(description="Create spinoffs of a baseline config with certain parameters modified")
    parser.add_argument("--data", type=str, help="Folder of offline reconstructions", default="archidata/adapted_dtw_v1.npy")
    parser.add_argument("--drop_bad", action='store_true', help="Drop high error exemplars")
    parser.add_argument("--version", type=int, default=VERSION, help="Drop high error exemplars")

    # archidata/adapted_dtw_v2.npy
    if True:
        import shlex
        narg1 = f"--drop_bad --data {DATA}"
        narg2 = f"--data {DATA}"
        for narg in narg1,: #narg2:
            args = parser.parse_args(shlex.split(narg))
            main(args)
    else:
        args = parser.parse_args()
        main(args)
    ## CHECK IF WIDTH IS TOO LONG
    ## MAKE A VARIANT WHERE YOU FILTER HIGH ERROR ONES