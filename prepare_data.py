from __future__ import print_function
import time, multiprocessing
from pathlib import Path
import os
from xml.etree import ElementTree
from tqdm import tqdm
import numpy as np
import utils
import drawing


def get_stroke_sequence(filename):
    tree = ElementTree.parse(filename).getroot()
    strokes = [i for i in tree if i.tag == 'StrokeSet'][0]

    coords = []
    for stroke in strokes:
        for i, point in enumerate(stroke):
            coords.append([
                int(point.attrib['x']),
                -1*int(point.attrib['y']),
                int(i == len(stroke) - 1)
            ])
    coords = np.array(coords)
    coords = drawing.align(coords)
    coords = drawing.denoise(coords)
    offsets = drawing.coords_to_offsets(coords)
    #offsets = offsets[:drawing.MAX_STROKE_LEN]
    offsets = drawing.normalize(offsets)

    return coords, offsets


def get_ascii_sequences(filename):
    sequences = open(filename, 'r').read()
    sequences = sequences.replace(r'%%%%%%%%%%%', '\n')
    sequences = [i.strip() for i in sequences.split('\n')]
    lines = sequences[sequences.index('CSR:') + 2:]
    lines = [line.strip() for line in lines if line.strip()]
    ascii_lines = [drawing.encode_ascii(line)[:drawing.MAX_CHAR_LEN] for line in lines]
    return ascii_lines, lines

def process(fname):
    # print(i, fname)
    if fname == 'data/raw/ascii/z01/z01-000/z01-000z.txt':
        return None

    head, tail = os.path.split(fname)
    last_letter = os.path.splitext(fname)[0][-1]
    last_letter = last_letter if last_letter.isalpha() else ''

    line_stroke_dir = head.replace('ascii', 'lineStrokes')
    line_stroke_fname_prefix = os.path.split(head)[-1] + last_letter + '-'
    # print(line_stroke_dir)
    if not os.path.isdir(line_stroke_dir):
        return None

    line_stroke_fnames = sorted([f for f in os.listdir(line_stroke_dir)
                                 if f.startswith(line_stroke_fname_prefix)])
    if not line_stroke_fnames:
        return None

    original_dir = head.replace('ascii', 'original')
    original_xml = os.path.join(original_dir, 'strokes' + last_letter + '.xml')
    tree = ElementTree.parse(original_xml)
    root = tree.getroot()

    general = root.find('General')
    if general is not None:
        writer_id = int(general[0].attrib.get('writerID', '0'))
    else:
        writer_id = int('0')

    ascii_sequences, text_group = get_ascii_sequences(fname)
    assert len(ascii_sequences) == len(line_stroke_fnames)
    return {"line_stroke_dir":line_stroke_dir,
            "ascii_sequences":ascii_sequences,
            "line_stroke_fnames":line_stroke_fnames,
            "text_group":text_group,
            "writer_id":writer_id}


def collect_data():
    global counter
    fnames = []
    for dirpath, dirnames, filenames in os.walk('data/raw/ascii/'):
        if dirnames:
            continue
        for filename in filenames:
            if filename.startswith('.'):
                continue
            fnames.append(os.path.join(dirpath, filename))

    # low quality samples (selected by collecting samples to
    # which the trained model assigned very low likelihood)
    blacklist = set(np.load('data/blacklist.npy', allow_pickle=True))

    stroke_fnames, transcriptions, writer_ids, texts = [], [], [], []
    counter = 0

    def callback(result):
        global counter
        counter += 1
        if result is None:
            return
        line_stroke_dir, ascii_sequences, line_stroke_fnames, text_group, writer_id = \
            result["line_stroke_dir"], result["ascii_sequences"], result["line_stroke_fnames"], result["text_group"], result["writer_id"]
        for ascii_seq, line_stroke_fname, text in zip(ascii_sequences, line_stroke_fnames, text_group):
            if line_stroke_fname in blacklist:
                continue

            stroke_fnames.append(os.path.join(line_stroke_dir, line_stroke_fname))
            transcriptions.append(ascii_seq)
            writer_ids.append(writer_id)
            texts.append(text)
        assert len(texts) == len(stroke_fnames)

    poolcount = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=poolcount)

    for i, fname in enumerate(tqdm(fnames)):
        if False:
            callback(process(fname))
        else:
            pool.apply_async(func=process, args=(fname,), callback=callback)

    pool.close()

    previous = 0
    with tqdm(total=len(fnames)) as pbar:
        while previous < len(fnames):
            time.sleep(1)
            new = counter
            pbar.update(new - previous)
            previous = new
    pool.join()

    return stroke_fnames, transcriptions, writer_ids, texts

def main():
    print('traversing data directory...')
    stroke_fnames, transcriptions, writer_ids, text = collect_data()

    print('dumping to numpy arrays...')
    x = np.zeros([len(stroke_fnames), drawing.MAX_STROKE_LEN, 3], dtype=np.float32)
    x_len = np.zeros([len(stroke_fnames)], dtype=np.int16)
    c = np.zeros([len(stroke_fnames), drawing.MAX_CHAR_LEN], dtype=np.int8)
    c_len = np.zeros([len(stroke_fnames)], dtype=np.int8)
    w_id = np.zeros([len(stroke_fnames)], dtype=np.int16)
    valid_mask = np.zeros([len(stroke_fnames)], dtype=np.bool)

    for i, (stroke_fname, c_i, w_id_i) in enumerate(tqdm(zip(stroke_fnames, transcriptions, writer_ids), total=len(stroke_fnames))):
        coords, offsets = get_stroke_sequence(stroke_fname)
        x_i = offsets

        # Exclude long strokes
        valid_mask[i] = ~np.any(np.linalg.norm(x_i[:, :2], axis=1) > 60) # exclude where stroke distance bigger than 60

        # Exclude wide images
        if len(offsets) > drawing.MAX_STROKE_LEN:
            valid_mask[i] = 0
            continue

        x[i, :len(x_i), :] = x_i
        x_len[i] = len(x_i)

        c[i, :len(c_i)] = c_i
        c_len[i] = len(c_i)

        w_id[i] = w_id_i

    output = Path('data/processed/original')
    output.mkdir(exist_ok=True, parents=True)

    np.save(output / 'x.npy', x[valid_mask])
    np.save(output / 'x_len.npy', x_len[valid_mask])
    np.save(output / 'c.npy', c[valid_mask])
    np.save(output / 'c_len.npy', c_len[valid_mask])
    np.save(output / 'w_id.npy', w_id[valid_mask])
    np.save(output / 'text.npy', [t for i,t in enumerate(text) if valid_mask[i]])

def combine(drop_bad=True):
    var = "processed" if not drop_bad else "processed_drop_bad"
    original = Path(f"data/{var}/original")
    new = Path(f"data/{var}")
    output = Path(f"data/{var}/processed_combined")

    #for file in ['x.npy', "x_len.npy", 'c.npy', 'c_len.npy', 'text.npy']: #'w_id.npy',
    x = 0
    for file in ['x.npy', "x_len.npy", 'c.npy', 'c_len.npy', 'text.npy', 'w_id.npy']:
        comb = np.concatenate([np.load(original / file, allow_pickle=True), np.load(new / file, allow_pickle=True)], axis=0)
        np.save(output / file, comb)

        if x:
            assert len(comb) == x
        else:
            x = len(comb)

def load():
    from matplotlib import pyplot as plt

    original = Path("data/processed/original")
    new = Path("data/processed")
    combined = Path("data/processed/processed_combined")

    offsets = np.load(combined / "x.npy", allow_pickle=True)
    utils.plot_from_synth_format(offsets[0])
    utils.plot_from_synth_format(offsets[-1])

    for axis in 0,1:
        print(f"AXIS {axis}")
        other = offsets[0:1000][:,:,axis].flatten()
        print(other.shape)

        plt.hist(other, range=(-5,5))
        plt.show()

        other = offsets[-1000:][:,:,axis].flatten()
        print(other.shape)
        plt.hist(other, range=(-5,5))
        plt.show()


if __name__ == '__main__':
    main()
    combine(drop_bad=True)
    combine(drop_bad=False)
    load()

    # x = np.load(Path("data/processed/original/text.npy"), allow_pickle=True)
    # print(x.shape)
    # y = np.load(Path("data/processed/original/x_len.npy"), allow_pickle=True)
    # print(y.shape)
    # print(x)