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


if __name__ == "__main__":
    hand = Hand()
    # usage demo
    for i in range(10):
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
        with open(f'train_synth_{i}.json', 'w') as fp:
            json.dump(data, fp)
