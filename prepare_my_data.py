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
from prepare_synth_data import *
import json
"""
Takes strokes recovered from offline data and prepares them to be used as training data

"""


def load_json_data(data_paths, root="", images_to_load=None):
    """

    Args:
        root (str):
        data_paths (list): list of str etc.
        images_to_load (int): how many to load

    Returns:

    """
    data = []
    root = Path(root)
    for data_path in data_paths:
        print("Loading JSON: ", data_path)
        with (root / data_path).open(mode="r") as fp:
            new_data = json.load(fp)
            if isinstance(new_data, dict):
                new_data = [item for key, item in new_data.items()]
            # print(new_data[:100])
            data.extend(new_data)

    if images_to_load:
        data = data[:images_to_load]
    print("Dataloader size", len(data))
    return data

def load_all_gts(gt_path):
    global GT_DATA
    data = load_json_data(data_paths=gt_path.glob("*.json"))
    #{'gt': 'He rose from his breakfast-nook bench', 'image_path': 'prepare_IAM_Lines/lines/m01/m01-049/m01-049-00.png',
    GT_DATA = {}
    for i in data:
        key = Path(i["image_path"]).stem.lower()
        assert not key in GT_DATA
        GT_DATA[key] = i["gt"]
    #print(f"GT's found: {GT_DATA.keys()}")
    return GT_DATA

def main(args):
    root = get_folder()
    output_path = "data/processed/original_mine"
    (Path(root) / output_path).mkdir(exist_ok=True, parents=True)

    data = load_data(args.data)
    output = []
    gt_lookup = np.load("data/processed/text_easy.npy", allow_pickle=True).item()

    for i,item in enumerate(data):
        if "_" in Path(item["image_path"]).name:
            id = Path(item["image_path"]).stem
            id = id[:id.find("_")]
            if id in gt_lookup:
                item.update({"id": id, "text": gt_lookup[id], "stroke":item["gt"], "distance":0})
                output.append(item)
    print(len(output))
    strokes, chars, w_id, text = process_data(output, drop_bad=False, parallel=True)

    x = np.zeros([len(strokes), drawing.MAX_STROKE_LEN, 3], dtype=np.float32) # BATCH, WIDTH, (X,Y,EOS)
    x_len = np.zeros([len(strokes)], dtype=np.int16)
    c = np.zeros([len(strokes), drawing.MAX_CHAR_LEN], dtype=np.int8)
    c_len = np.zeros([len(strokes)], dtype=np.int8)

    for i, (stroke, char) in tqdm(enumerate(zip(strokes, chars))):
        x[i, :len(stroke), :] = stroke
        x_len[i] = len(stroke)

        c[i, :len(char)] = char
        c_len[i] = len(char)

    np.save(root+f'{output_path}/x.npy', x)
    np.save(root+f'{output_path}/text.npy', text)
    np.save(root+f'{output_path}/x_len.npy', x_len)
    np.save(root+f'{output_path}/c.npy', c)
    np.save(root+f'{output_path}/c_len.npy', c_len)
    np.save(root+f'{output_path}/w_id.npy', w_id)
    np.save(root+f'{output_path}/text.npy', text)
    print("Valid strokes", len(x))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create spinoffs of a baseline config with certain parameters modified")
    parser.add_argument("--data", type=str, help="Folder of offline reconstructions", default="/media/data/GitHub/simple_hwr/RESULTS/pretrained/adapted_v2/training_dataset.npy")

    if False:
        import shlex
        nargs = "--data archidata/adapted_dtw_v1.npy"
        args = parser.parse_args(shlex.split(nargs))
    else:
        args = parser.parse_args()
    main(args)
    ## CHECK IF WIDTH IS TOO LONG
    ## MAKE A VARIANT WHERE YOU FILTER HIGH ERROR ONES