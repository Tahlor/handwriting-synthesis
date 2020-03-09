import demo
from tqdm import tqdm
CHECKPOINT = "checkpoints/original"

from demo import *
from drawing import alphabet
import json

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


punc = [";",":",",",".","!",'"',"'"]
char_set = ["i","i","i","I","t","t","t","T","F","H","K","f",'E',"A","B","J","o","p","g","y","s","S","x"]
def get_invented_line():
    line = ""
    line_length = np.random.randint(14, 35)

    while len(line) < line_length:
        word_length = np.random.randint(2,7)
        word = ""
        for letter in range(word_length):
            word += char_set[np.random.randint(0,len(char_set))]

        # Punctuation
        if np.random.randint(0,4)==0:
            word += punc[np.random.randint(0, len(punc))]
        line += word + " "
    return line


def process():
    hand = Hand(checkpoint="checkpoints/original")
    # usage demo
    vers = "random"
    for i in range(5):
        if vers=="random":
            lines = [get_invented_line() for n in tqdm(range(10000))]
        else:
            lines = get_lines(n=10000)

        biases = list(np.random.rand(len(lines)))
        styles = list(np.random.randint(13, size=len(lines)))

        strokes = hand.write(
            filename='img/test.svg',
            lines=lines,
            biases=biases,
            styles=styles,
            return_strokes=True,
            only_strokes=True
        )
        data = [{'text': line, 'stroke': stroke.tolist()} for line, stroke in zip(lines, strokes)]
        with open(f'train_synth_{vers}_{i}.json', 'w') as fp:
            json.dump(data, fp)

if __name__ == "__main__":
    print(get_invented_line())
    process()
