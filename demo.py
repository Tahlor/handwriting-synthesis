from utils import *
import os
import logging
from tqdm import tqdm
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import svgwrite

import drawing
import lyrics
from rnn import rnn
from create_style import load_samples

import os
CHECKPOINT = 'checkpoints/original'

class Hand(object):

    def __init__(self, checkpoint=CHECKPOINT):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.nn = rnn(
            log_dir=f'{checkpoint}/logs',
            checkpoint_dir=checkpoint,
            prediction_dir='predictions',
            learning_rates=[.0001, .00005, .00002],
            batch_sizes=[32, 64, 64],
            patiences=[1500, 1000, 500],
            beta1_decays=[.9, .9, .9],
            validation_batch_size=32,
            optimizer='rms',
            num_training_steps=100000,
            warm_start_init_step=17900,
            regularization_constant=0.0,
            keep_prob=1.0,
            enable_parameter_averaging=False,
            min_steps_to_checkpoint=2000,
            log_interval=20,
            logging_level=logging.CRITICAL,
            grad_clip=10,
            lstm_size=400,
            output_mixture_components=20,
            attention_mixture_components=10
        )
        self.checkpoint_dir = Path(checkpoint)
        self.img_dir = (self.checkpoint_dir / "img")
        self.img_dir.mkdir(exist_ok=True, parents=True)
        print(f"Loading from {checkpoint}")
        self.nn.restore()

    @staticmethod
    def validate_line(line, line_num, valid_char_set):
        if len(line) > 75:
            raise ValueError(
                (
                    "Each line must be at most 75 characters. "
                    "Line {} contains {}"
                ).format(line_num, len(line))
            )

        for char in line:
            if char not in valid_char_set:
                raise ValueError(
                    (
                        "Invalid character {} detected in line {}. "
                        "Valid character set is {}"
                    ).format(char, line_num, valid_char_set)
                )


    def write(self, filename, lines, biases=None, styles=None, stroke_colors=None, stroke_widths=None, draw=True):
        valid_char_set = set(drawing.alphabet)
        for line_num, line in enumerate(lines):
            self.validate_line(line, line_num, valid_char_set)

        strokes = self._sample(lines, biases=biases, styles=styles)
        if draw:
            self._draw(strokes, lines, (self.img_dir / filename).as_posix(), stroke_colors=stroke_colors, stroke_widths=stroke_widths)
        else:
            # Just draw one sample two
            self._draw(strokes[:2], lines[:2], (self.img_dir / filename).as_posix())
        final_strokes = list(self._finalize_strokes(strokes, lines))
        return final_strokes

    def _sample(self, lines, biases=None, styles=None):
        num_samples = len(lines)
        max_tsteps = 40*max([len(i) for i in lines])
        biases = biases if biases is not None else [0.5]*num_samples

        x_prime = np.zeros([num_samples, 1200, 3])
        x_prime_len = np.zeros([num_samples])
        chars = np.zeros([num_samples, 120])
        chars_len = np.zeros([num_samples])

        if styles is not None:
            for i, (cs, style) in enumerate(zip(lines, styles)):
                if isinstance(style, dict):
                    x_p = style['stroke']
                    c_p = style['text']
                else:
                    x_p = np.load('styles/style-{}-strokes.npy'.format(style))
                    c_p = np.load('styles/style-{}-chars.npy'.format(style)).tostring().decode('utf-8')

                c_p = str(c_p) + " " + cs
                c_p = drawing.encode_ascii(c_p)
                c_p = np.array(c_p)

                x_prime[i, :len(x_p), :] = x_p
                x_prime_len[i] = len(x_p)
                chars[i, :len(c_p)] = c_p
                chars_len[i] = len(c_p)

        else:
            for i in range(num_samples):
                encoded = drawing.encode_ascii(lines[i])
                chars[i, :len(encoded)] = encoded
                chars_len[i] = len(encoded)

        [samples] = self.nn.session.run(
            [self.nn.sampled_sequence],
            feed_dict={
                self.nn.prime: styles is not None,
                self.nn.x_prime: x_prime,
                self.nn.x_prime_len: x_prime_len,
                self.nn.num_samples: num_samples,
                self.nn.sample_tsteps: max_tsteps,
                self.nn.c: chars,
                self.nn.c_len: chars_len,
                self.nn.bias: biases
            }
        )
        samples = [sample[~np.all(sample == 0.0, axis=1)] for sample in samples]
        return samples

    def _finalize_strokes(self, strokes, lines=None):
        """
        Lines just used to determine if line is trivial/vacuous
        """
        for i, offsets in tqdm(enumerate(strokes)):
            if lines and not lines[i]:
                print("Empty line? Stroke:")
                print(offsets[:10])
                continue

            offsets[:, :2] *= 1.5
            curr_strokes = drawing.offsets_to_coords(offsets)
            curr_strokes = drawing.denoise(curr_strokes)
            curr_strokes[:, :2] = drawing.align(curr_strokes[:, :2])

            # Normalize
            curr_strokes[:, 1] -= np.min(curr_strokes[:, 1])
            max_y = np.max(curr_strokes[:, 1])
            if max_y:
                curr_strokes[:, :2] /= np.max(curr_strokes[:, 1])
            else:
                warnings.warn(f"max y is zero {curr_strokes}")

            # Convert end points to start points
            #curr_strokes = eos_to_sos(curr_strokes)

            yield curr_strokes

    def _draw(self, strokes, lines, filename, stroke_colors=None, stroke_widths=None):
        stroke_colors = stroke_colors or ['black']*len(lines)
        stroke_widths = stroke_widths or [2]*len(lines)

        line_height = 60
        view_width = 1000
        view_height = line_height*(len(strokes) + 1)

        dwg = svgwrite.Drawing(filename=filename)
        dwg.viewbox(width=view_width, height=view_height)
        dwg.add(dwg.rect(insert=(0, 0), size=(view_width, view_height), fill='white'))

        initial_coord = np.array([0, -(3*line_height / 4)])
        for i, (offsets, line, color, width) in tqdm(enumerate(zip(strokes, lines, stroke_colors, stroke_widths))):

            if not line: # insert return character
                initial_coord[1] -= line_height
                continue

            offsets[:, :2] *= 1.5
            curr_strokes = drawing.offsets_to_coords(offsets)
            curr_strokes = drawing.denoise(curr_strokes)
            curr_strokes[:, :2] = drawing.align(curr_strokes[:, :2])


            curr_strokes[:, 1] *= -1
            curr_strokes[:, :2] -= curr_strokes[:, :2].min() + initial_coord
            curr_strokes[:, 0] += (view_width - curr_strokes[:, 0].max()) / 2

            prev_eos = 1.0
            p = "M{},{} ".format(0, 0)
            for x, y, eos in zip(*curr_strokes.T):
                p += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x, y)
                prev_eos = eos
            path = svgwrite.path.Path(p)
            path = path.stroke(color=color, width=width, linecap='round').fill("none")
            dwg.add(path)

            initial_coord[1] -= line_height
        dwg.save()
        return strokes


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    import re, warnings

    parser = argparse.ArgumentParser(description="Create spinoffs of a baseline config with certain parameters modified")
    parser.add_argument("--checkpoint_folder", type=str, help="Folder of checkpoints", default='checkpoints/original')
    args = parser.parse_args()

    hand = Hand(args.checkpoint_folder)

    # usage demo
    lines = [
        "Now this is a story all about how",
        "My life got flipped turned upside down",
        "And I'd like to take a minute, just sit right there",
        "I'll tell you how I became the prince of a town called Bel-Air",
    ]
    biases = [.75 for i in lines]
    styles = [9 for i in lines]
    stroke_colors = ['red', 'green', 'black', 'blue']
    stroke_widths = [1, 2, 1, 2]

    print("Usage demo...")
    hand.write(
        filename='usage_demo.svg',
        lines=lines,
        biases=biases,
        styles=styles,
        stroke_colors=stroke_colors,
        stroke_widths=stroke_widths
    )

    # demo number 1 - fixed bias, fixed style
    lines = lyrics.all_star.split("\n")
    biases = [.75 for i in lines]
    styles = [12 for i in lines]

    print("All star...")
    hand.write(
        filename='all_star.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )

    print("Thousand Miles...")
    # demo number 2 - fixed bias, varying style
    lines = lyrics.downtown.split("\n")
    biases = [.75 for i in lines]
    styles = np.cumsum(np.array([len(i) for i in lines]) == 0).astype(int)

    hand.write(
        filename='downtown.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )

    print("Never gonna give you up...")
    # demo number 3 - varying bias, fixed style
    lines = lyrics.give_up.split("\n")
    biases = .2*np.flip(np.cumsum([len(i) == 0 for i in lines]), 0)
    styles = [7 for i in lines]

    hand.write(
        filename='give_up.svg',
        lines=lines,
        biases=biases,
        styles=styles,
    )
