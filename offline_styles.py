# Load reconstructed strokes etc.
# Load writers too
import numpy as np
from pathlib import Path
from utils import *
import re
from collections import defaultdict


root = Path("/media/data/GitHub/simple_hwr/")
data_path = root/"RESULTS/OFFLINE_PREDS/good/imgs/current/eval/data/all_data.npy"

data = np.load(data_path, allow_pickle=True)

# Author
output_dict = defaultdict(list)
for i in data:
    author = "-".join(i["id"].split("-")[:2])
    output_dict[author].append(i)

all_styles = []
for author, stroke_dict_list in output_dict.items():
    for stroke_dict in stroke_dict_list:
        new_stroke = convert_gts_to_synth_format(stroke_dict["stroke"])
        all_styles.append({"stroke":new_stroke, "text":stroke_dict["text"], "author":author})
        continue # just take one sample from each author

np.save("styles/all_offline_styles.npy", all_styles)
np.save("styles/sample_offline_styles.npy", all_styles[:100])