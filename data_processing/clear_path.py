from pathlib import Path
import shutil
import os, sys
import argparse
root = Path(os.path.dirname(os.path.realpath(__file__))).parent
sys.path.append(str(root))
import utils

PATH = "/panfs/pan.fsl.byu.edu/scr/grp/fslg_hwr/handwriting-synthesis/checkpoints/V4.1"

def clean_folder(args):
    PATH = args.folder
    for extension in "*.npy", "*.log", "*.slurm":
        for f in Path(PATH).rglob(extension):
            os.remove(f)

    # Remove folders
    for f in Path(PATH).rglob("*"):
        if f.is_dir():
            if f.name in ["img", "logs"]:
                shutil.rmtree(f)
        if f.name == "checkpoint":
            # Update checkpoint with most recent one available
            max_checkpoint = utils.get_max_checkpoint(f.parent)
            if max_checkpoint:
                with f.open("w") as _f:
                    _f.write("\n".join([x.strip() for x in f"""
                    model_checkpoint_path: "{f.parent}/model-{max_checkpoint}"
                    all_model_checkpoint_paths: "{f.parent}/model-{max_checkpoint}"
                    """.split("\n")]))
            else:
                os.remove(f)

if __name__=='__main__':
    import shlex
    parser = argparse.ArgumentParser(description="Create spinoffs of a baseline config with certain parameters modified")
    parser.add_argument("--folder", type=str, help="Folder of checkpoints", default=PATH)
    args = parser.parse_args()

    really = input(f"Really delete all .npy, .log, img, and log files/folders in {args.folder}? Y/n ")
    if really.lower() == "y":
        clean_folder(args)