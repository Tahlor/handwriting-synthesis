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
output_dict = defaultdict(int)
for i in data:
    author = "-".join(i["id"].split("-")[:2])
    output_dict[author].append(i)

for author in output_dict.keys():
    pass


