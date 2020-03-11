from PIL import Image, ImageDraw
from math import ceil
import warnings, re
from pathlib import Path
import os
import socket
import numpy as np

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


def draw_from_gt(gt, show=True, save_path=None, min_width=None, height=61,
                 right_padding="random", linewidth=None, max_width=5, color=0, alpha=False,
                 use_stroke_number=None, plot_points=False):
    """ RETURNS DATA IN "LOWER" origin format!!!
        GT is a WIDTH x VOCAB size numpy array
        Start strokes are inferred by [:,2], which should be 1 when the point starts a new stroke
        [:,0:2] are the x,y coordinates

    Args:
        raw:
        save_path:
        height:
        use_stroke_number: True - strokes labelled like 1,1,1,1,2,2,2,2; False - 00000100001

    Returns:

    """
    ### HACK
    if use_stroke_number is None:
        use_stroke_number = True if gt.shape[-1] > 2 and np.any(gt[:,2] >= 2) else False

    if isinstance(color, int):
        color = color,
    else:
        color = tuple(color)
    channels = len(color)
    image_type = "L" if channels == 1 else "RGB"
    background = tuple([255]*channels)
    if alpha:
        image_type += "A"
        color = tuple((*color, 255))
        background = tuple((*background, 0))

    if linewidth is None:
        linewidth = min(max(int(abs(np.random.randn()) * (max_width - 1) * .5 + 1), 1),max_width)

    if isinstance(right_padding, str):
        right_padding = np.random.randint(6)

    #
    width = ceil(np.max(gt[:, 0]) * height) + right_padding
    width = max(width, height) # needs to be positive
    rescale = height

    if min_width:
        width = max(width, min_width)

    # else: # If a width is specified, we can't rescale to height
    #     max_rescale = min_width / np.max(gt[:, 0])
    #     rescale = min(height, max_rescale)

    gt_rescaled = np.c_[gt[:, 0:2] * rescale, gt[:, 2:]]
    pil_format = gt_to_pil_format(gt_rescaled, stroke_number=use_stroke_number)
    img = Image.new(image_type, (width, height), background)
    draw = ImageDraw.Draw(img)

    for line in pil_format:
        if line.size > 2:
            line = [tuple(x) for x in line.flatten().reshape(-1, 2).tolist()]
            draw.line(line, fill=color, width=linewidth, joint='curve')
        elif line.size == 2: # only have a single coordinate, make it big!
            line1 = line - linewidth / 2
            line2 = line + linewidth / 2
            line = np.r_[line1, line2].flatten().tolist()
            draw.ellipse(line, fill=color, outline=color)

    if plot_points:
        image_type = "RGB"
        stroke_point_size=2
        background = Image.new(image_type, (width, height), (255, 255, 255))
        background.paste(img)  # 3 is the alpha channel
        draw = ImageDraw.Draw(background)

        for line in pil_format:
            for i, point in enumerate(line):
                color = 'blue' if i else 'orange'
                line1 = point - stroke_point_size / 2
                line2 = point + stroke_point_size / 2
                point = np.r_[line1, line2].flatten().tolist()
                draw.ellipse(point, fill=color, outline=color)
        img = background

    data = np.array(img)[::-1]  # invert the y-axis, to upper origin

    img = Image.fromarray(data, image_type)

    if save_path:
        img.save(save_path)
    if show:
        img.show()

    return data

def eos_to_sos(line):
    line[1:, -1] = line[:-1, -1]
    line[0, -1] = 1
    return line

def sos_to_eos(line):
    line[:-1, -1] = line[1:, -1]
    line[-1,-1] = 1
    return line

def gt_to_pil_format(instance, stroke_number=True, has_start_points=True):
    """

    Args:
        instance: NUMPY!

    Returns:
        Pil format; list of strokes Length X (x,y)
    """

    has_start_points = False if instance.shape[-1] <= 2 else True

    if has_start_points:
        # if start points are sequential 000011112222...
        start_points = relativefy(instance[:, 2]) if stroke_number else instance[:, 2]
        start_indices = np.argwhere(np.round(start_points) == 1).astype(int).reshape(-1)
        l = np.split(instance[:, 0:2], start_indices)
        if np.any(start_points < 0):
            raise Exception("Start points are less than 0")
        return l
    else:
        one_liner = [instance.flatten()]
        return one_liner


if __name__=="__main__":
    print(project_root())
    print(get_folder("data"))
    print(get_max_checkpoint("./checkpoints/original"))