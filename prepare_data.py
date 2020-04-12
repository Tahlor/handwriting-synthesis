from __future__ import print_function
import time, multiprocessing
from pathlib import Path
import os
from xml.etree import ElementTree
from tqdm import tqdm
import numpy as np
import utils
import drawing
import shutil

# 568 - 'data/raw/lineStrokes/a01/a01-000/a01-000u-01.xml'
def get_stroke_sequence(filename, resample=False, first_stroke_0=False):
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

    # coords2 = drawing.align(coords.copy())
    # coords2 = drawing.denoise(coords)
    # offsets2 = drawing.coords_to_offsets(coords2)
    # #offsets = offsets[:drawing.MAX_STROKE_LEN] These are excluded later, truncating them would prevent exclusion
    # offsets2 = drawing.normalize(offsets2)

    if resample:
        coords = utils.resample_coords(coords)

    coords = drawing.align(coords)
    coords = drawing.denoise(coords)
    offsets = drawing.coords_to_offsets(coords, first_stroke_0=first_stroke_0)
    #offsets = offsets[:drawing.MAX_STROKE_LEN] These are excluded later, truncating them would prevent exclusion
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
        line_stroke_dir, ascii_sequences, line_stroke_fnames, text_group, writer_id, fnames = \
            result["line_stroke_dir"], result["ascii_sequences"], result["line_stroke_fnames"], \
            result["text_group"], result["writer_id"], result["path"]
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
            "writer_id":writer_id,
            "path":fname}



def main(args):
    print('traversing data directory...')
    stroke_fnames, transcriptions, writer_ids, text = collect_data()

    print('dumping to numpy arrays...')
    x = np.zeros([len(stroke_fnames), drawing.MAX_STROKE_LEN, 3], dtype=np.float32)
    x_len = np.zeros([len(stroke_fnames)], dtype=np.int16)
    c = np.zeros([len(stroke_fnames), drawing.MAX_CHAR_LEN], dtype=np.int8)
    c_len = np.zeros([len(stroke_fnames)], dtype=np.int8)
    w_id = np.zeros([len(stroke_fnames)], dtype=np.int16)
    valid_mask = np.zeros([len(stroke_fnames)], dtype=np.bool)

    ii = 0

    for i, (stroke_fname, c_i, w_id_i) in enumerate(tqdm(zip(stroke_fnames, transcriptions, writer_ids), total=len(stroke_fnames))):
        coords, offsets = get_stroke_sequence(stroke_fname, resample=False, first_stroke_0=args.first_stroke_0)
        x_i = offsets
        #coords, offsets = get_stroke_sequence(stroke_fname, resample=False)

        # Exclude long strokes
        valid_mask[i] = ~np.any(np.linalg.norm(x_i[:, :2], axis=1) > 60) # exclude where stroke distance bigger than 60

        # Exclude wide images
        if len(offsets) > drawing.MAX_STROKE_LEN:
            ii += 1
            print(ii)
            valid_mask[i] = 0
            continue

        x[i, :len(x_i), :] = x_i
        x_len[i] = len(x_i)

        c[i, :len(c_i)] = c_i
        c_len[i] = len(c_i)

        w_id[i] = w_id_i
        # if i > 10:
        #     break

    if args.first_stroke_0:
        output = Path('data/processed/original0')
        assert x[0,0,2]==0
    else:
        output = Path('data/processed/original')
        assert x[0, 0, 2] == 1

    output.mkdir(exist_ok=True, parents=True)

    np.save(output / 'x.npy', x[valid_mask])
    np.save(output / 'x_len.npy', x_len[valid_mask])
    np.save(output / 'c.npy', c[valid_mask])
    np.save(output / 'c_len.npy', c_len[valid_mask])
    np.save(output / 'w_id.npy', w_id[valid_mask])
    np.save(output / 'text.npy', [t for i,t in enumerate(text) if valid_mask[i]])


    output_dict = {}
    for i,t in enumerate(text):
        output_dict[Path(stroke_fnames[i]).stem] = t
    np.save(output / 'text_easy.npy', output_dict)
    shutil.copy(output, Path("/synth") / output )

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Create spinoffs of a baseline config with certain parameters modified")
    parser.add_argument("--first_stroke_0", action="store_true", help="Folder of offline reconstructions", default=False)
    args = parser.parse_args()
    main(args)

    # x = np.load(Path("data/processed/original/text.npy"), allow_pickle=True)
    # print(x.shape)
    # y = np.load(Path("data/processed/original/x_len.npy"), allow_pickle=True)
    # print(y.shape)
    # print(x)