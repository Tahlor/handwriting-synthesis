from __future__ import print_function
from pathlib import Path
import os
from xml.etree import ElementTree
from tqdm import tqdm
import multiprocessing
import numpy as np
import time
import sys

LOCAL = Path(os.path.dirname(os.path.realpath(__file__)))
ROOT = LOCAL.parent

sys.path.extend([ROOT,LOCAL])
import drawing2 as drawing
#import drawing

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
    offsets = offsets[:drawing.MAX_STROKE_LEN]
    offsets = drawing.normalize(offsets)
    return offsets


def get_ascii_sequences(filename):
    sequences = open(filename, 'r').read()
    sequences = sequences.replace(r'%%%%%%%%%%%', '\n')
    sequences = [i.strip() for i in sequences.split('\n')]
    lines = sequences[sequences.index('CSR:') + 2:]
    lines = [line.strip() for line in lines if line.strip()]
    lines = [drawing.encode_ascii(line)[:drawing.MAX_CHAR_LEN] for line in lines]
    return lines

def collect_data():
    fnames = []
    for dirpath, dirnames, filenames in os.walk(ROOT / 'data/raw/ascii/'):
        if dirnames:
            continue
        for filename in filenames:
            if filename.startswith('.'):
                continue
            fnames.append(os.path.join(dirpath, filename))

    # low quality samples (selected by collecting samples to
    # which the trained model assigned very low likelihood)
    blacklist = set(np.load(ROOT / 'data/blacklist.npy', allow_pickle=True))

    stroke_fnames, transcriptions, writer_ids = [], [], []
    for i, fname in enumerate(fnames):
        print(i, fname)
        if fname == str(ROOT / 'data/raw/ascii/z01/z01-000/z01-000z.txt'):
            continue

        head, tail = os.path.split(fname)
        last_letter = os.path.splitext(fname)[0][-1]
        last_letter = last_letter if last_letter.isalpha() else ''

        line_stroke_dir = head.replace('ascii', 'lineStrokes')
        line_stroke_fname_prefix = os.path.split(head)[-1] + last_letter + '-'

        if not os.path.isdir(line_stroke_dir):
            continue
        line_stroke_fnames = sorted([f for f in os.listdir(line_stroke_dir)
                                     if f.startswith(line_stroke_fname_prefix)])
        if not line_stroke_fnames:
            continue

        original_dir = head.replace('ascii', 'original')
        original_xml = os.path.join(original_dir, 'strokes' + last_letter + '.xml')
        tree = ElementTree.parse(original_xml)
        root = tree.getroot()

        general = root.find('General')
        if general is not None:
            writer_id = int(general[0].attrib.get('writerID', '0'))
        else:
            writer_id = int('0')

        ascii_sequences = get_ascii_sequences(fname)
        assert len(ascii_sequences) == len(line_stroke_fnames)

        for ascii_seq, line_stroke_fname in zip(ascii_sequences, line_stroke_fnames):
            if line_stroke_fname in blacklist:
                continue

            stroke_fnames.append(os.path.join(line_stroke_dir, line_stroke_fname))
            transcriptions.append(ascii_seq)
            writer_ids.append(writer_id)

    return stroke_fnames, transcriptions, writer_ids

def process(fname):
    # print(i, fname)
    if fname == str(ROOT / 'data/raw/ascii/z01/z01-000/z01-000z.txt'):
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


if __name__ == '__main__':
    print('traversing data directory...')
    stroke_fnames, transcriptions, writer_ids = collect_data()
    output_folder = ROOT / "data/processed/original_original"
    print('dumping to numpy arrays...')
    x = np.zeros([len(stroke_fnames), drawing.MAX_STROKE_LEN, 3], dtype=np.float32)
    x_len = np.zeros([len(stroke_fnames)], dtype=np.int16)
    c = np.zeros([len(stroke_fnames), drawing.MAX_CHAR_LEN], dtype=np.int8)
    c_len = np.zeros([len(stroke_fnames)], dtype=np.int8)
    w_id = np.zeros([len(stroke_fnames)], dtype=np.int16)
    valid_mask = np.zeros([len(stroke_fnames)], dtype=np.bool)

    for i, (stroke_fname, c_i, w_id_i) in enumerate(zip(stroke_fnames, transcriptions, writer_ids)):
        if i % 200 == 0:
            print(i, '\t', '/', len(stroke_fnames))
        x_i = get_stroke_sequence(stroke_fname)

        if x_i is None or len(x_i) > drawing.MAX_STROKE_LEN:
            valid_mask[i] = 0
            continue
        else:
            valid_mask[i] = ~np.any(np.linalg.norm(x_i[:, :2], axis=1) > 60)

        x[i, :len(x_i), :] = x_i
        x_len[i] = len(x_i)

        c[i, :len(c_i)] = c_i
        c_len[i] = len(c_i)

        w_id[i] = w_id_i
        # if i > 500:
        #     break
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    np.save(Path(output_folder) / 'x.npy', x[valid_mask])
    np.save(Path(output_folder) / 'x_len.npy', x_len[valid_mask])
    np.save(Path(output_folder) / 'c.npy', c[valid_mask])
    np.save(Path(output_folder) / 'c_len.npy', c_len[valid_mask])
    np.save(Path(output_folder) / 'w_id.npy', w_id[valid_mask])

    output_dict = {}
    # for i,t in enumerate(text):
    #     output_dict[w_id[i]] = t
    # np.save(Path(output_folder) / 'text_easy.npy', output_dict)