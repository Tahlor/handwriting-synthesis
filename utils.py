import warnings, re
from pathlib import Path
import os
import socket

def get_computer():
    return socket.gethostname()

def is_galois():
    return get_computer() == "Galois"

def is_dalai():
    if get_computer() == "DalaiLama":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return True
    return False



def project_root(set_root=True):
    current_directory = Path(os.getcwd())

    while current_directory and current_directory.stem != "handwriting-synthesis":
        current_directory = current_directory.parent

    if set_root:
        os.chdir(current_directory.as_posix())

    return current_directory

def get_folder(folder="."):
    path = (project_root() / folder)
    if path.exists() and path.is_file():
        return path.as_posix()
    else:
        return (project_root() / folder).as_posix() + "/"

def get_max_checkpoint(checkpoint_folder):
    nums = []
    try:
        for c in Path(checkpoint_folder).glob("model-*"):
            matches = re.findall("([0-9]+)", c.stem)[0]
            if matches:
                nums.append(int(matches))
        warm_start = max(nums)
    except:
        warnings.warn("Couldn't find checkpoint")
        warm_start = 0
    return warm_start


if __name__=="__main__":
    print(project_root())
    print(get_folder("data"))
    print(get_max_checkpoint("./checkpoints/original"))