import argparse
from utils import *
import demo
from tqdm import tqdm
from demo import *
from drawing import alphabet
import json

CHECKPOINT = get_folder("./checkpoints/gen_training_data")

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_lines(path='raw_text_10000.txt', n=100000):
    with open(path, 'r') as f:
        text = f.read().split()
    n_text = len(text)
    lines = []
    print(max([len(w) for w in text]))
    while len(lines) < n:
        j = np.random.randint(n_text-10)
        words = []
        while True:
            line = ' '.join(words)
            if len(line) > np.random.randint(10, 30):
                if len(line) >= 75:
                    break
                lines.append(line)
                break
            else:
                word = text[j]
                if any(c not in alphabet for c in word):
                    break
                words.append(word)
                j += 1
    return lines

punc = [",",".","!",'"',"'", ";", ":"]
char_set = ["j","j","i","i","i","I","t","t","t","T","F","H","K","f",'E',"A","B","J","o","p","g","y","s","S","x","z"]*2
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
            'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
            'y', 'z']
char_set += alphabet

def capitalize(letter):
    # Capital
    if np.random.randint(0, 5) == 0 and letter.upper() in drawing.alphabet:
        return letter.upper()
    else:
        return letter

def get_invented_line():
    line = ""
    line_length = np.random.randint(14, 35)

    while len(line) < line_length:
        word_length = np.random.randint(1,10)
        word = ""

        for letter in range(word_length):
            word += capitalize(char_set[np.random.randint(0,len(char_set))])

        # Punctuation
        if np.random.randint(0,4)==0:
            word += punc[np.random.randint(0, len(punc))]

        line += word + " "
    return line


def process(vers="random", checkpoint=CHECKPOINT):
    # TESTING = True
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    batch_size = 1 if is_dalai() else 1000

    hand = Hand(checkpoint=checkpoint)
    # usage demo

    for i in range(100):
        number = 2 if i == 1 else batch_size
        if vers=="random":
            lines = [get_invented_line() for n in tqdm(range(number))]
        else:
            lines = get_lines(n=number)

        biases = list(np.random.rand(len(lines))*2)
        styles = list(np.random.randint(14, size=len(lines)))

        strokes = hand.write(
            filename=f'test_{vers}_{i}.svg',
            lines=lines,
            biases=biases,
            styles=styles,
            draw=False
        )

        data = [{'text': line, 'stroke': stroke.tolist(), 'bias': float(bias), 'style': float(style)} for line, stroke, bias, style in zip(lines, strokes, biases, styles)]

        with open(Path(CHECKPOINT) / f'train_synth_{vers}_{i}.json', 'w') as fp:
            json.dump(data, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create spinoffs of a baseline config with certain parameters modified")
    parser.add_argument("--checkpoint_folder", type=str, help="Folder of checkpoints", default=CHECKPOINT)
    parser.add_argument("--variant", type=str, help="'random' for random letters", default='normal')

    args = parser.parse_args()

    for i in range(10):
        print(get_invented_line())
    process(vers=args.variant, checkpoint=args.checkpoint_folder)
