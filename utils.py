import stroke_recovery
from scipy import interpolate
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from math import ceil
import warnings, re
from pathlib import Path
import os
import socket
import numpy as np
import signal
import drawing


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
    rel_path = "."
    while current_directory and current_directory.stem != "handwriting-synthesis":
        current_directory = current_directory.parent
        rel_path += "/.."

    if set_root:
        os.chdir(current_directory.as_posix())
        return Path(".")
    else:
        return Path(rel_path)

def get_folder(folder="."):
    path = (project_root() / folder)
    if path.exists() and path.is_file():
        return path.as_posix()
    else:
        return (project_root() / folder).as_posix() + "/"

def get_project_root():
    ROOT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
    while ROOT_DIR.name != "handwriting-synthesis":
        ROOT_DIR = ROOT_DIR.parent
    return ROOT_DIR


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


def draw_from_gt(gt, show=True, save_path=None, min_width=None, height=60,
                 right_padding="random", linewidth=None, max_width=5, color=0, alpha=False,
                 use_stroke_number=None, plot_points=False, **kwargs):
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
    line = line.copy()
    if np.asarray(line).ndim==2:
        #line[0,-1] = 0 # will now be the second keyframe
        line[1:, -1] = line[:-1, -1]
        line[0, -1] = 1
    elif np.asarray(line).ndim==3: # batch dimension
        line[:,1:, -1] = line[:,:-1, -1]
        line[:,0, -1] = 1
    else:
        line[1:] = line[:-1]
        line[0] = 1
    return line

def sos_to_eos(line):
    line = line.copy()

    if np.asarray(line).ndim==2:
        line[:-1, -1] = line[1:, -1]
        line[-1,-1] = 1
    elif np.asarray(line).ndim==3: # batch dimension
        line[:, :-1, -1] = line[:, 1:, -1]
        line[:, -1,-1] = 1
    else:
        line[:-1] = line[1:]
        line[-1] = 1
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

def is_taylor():
    return get_computer() in ("Galois", "brodie")

def convert_gts_to_synth_format(stroke, adjustments=True):
    new_stroke = stroke[:,:3].copy()
    if np.any(new_stroke[:,-1]>=2):
        #raise Exception("Input data is in stroke number format")
        warnings.warn("Input data is in stroke number format")
        new_stroke[:,-1] = stroke_recovery.relativefy(new_stroke[:,-1])
        assert not np.any(new_stroke[:,-1]>=2)
    # Round SOS
    new_stroke[:, -1] = np.round(new_stroke[:, -1])

    #if np.all(new_stroke[0,:2] != 0):
    #new_stroke = np.concatenate([np.array([[0,0,0]]), new_stroke], axis=0)

    # Convert to EOS
    coords = sos_to_eos(new_stroke)
    if adjustments:
        coords = drawing.align(coords)
        coords = drawing.denoise(coords)
    offsets = drawing.coords_to_offsets(coords)
    offsets = offsets[:drawing.MAX_STROKE_LEN]
    offsets = drawing.normalize(offsets)
    return offsets

def convert_synth_offsets_to_gt(offsets, return_all=False):
    test = offsets.copy()
    test[:, :2] = np.cumsum(test[:, :2], axis=0)
    xmin, ymin, factor = np.min(test[:, 0]), np.min(test[:, 1]), np.max(test[:, 1])-np.min(test[:, 1])
    test[:, 1] -= np.min(test[:, 1]) # min_y = 0
    test[:, 0] -= np.min(test[:, 0]) # min_x = 0
    if factor != 0:
        test[:, :2] /= factor
    test = eos_to_sos(test)
    #test = test[1:] # remove the first 0,0 point
    if return_all:
        return test, xmin, ymin, factor
    return test

def kill_gpu_hogs(force=False):
    if is_taylor():
        from subprocess import Popen, DEVNULL, STDOUT, check_output
        pid = os.getpid()
        exclusion_words = "visdom", "jupyter", "grep"
        find_processes_command = f"ps all | grep python"  + f" | awk '!/{pid}/'"
        x = check_output([find_processes_command], shell=True)
        all_python_processes = x.decode().split("\n")[:-1]
        for process in all_python_processes:
            if not any([ew in process.lower() for ew in exclusion_words]):
                try:
                    os.kill(int(process.split()[2]), signal.SIGTERM)
                except:
                    pass


def plot_from_synth_format(offsets, show=False, save_path=None, force=False, **kwargs):
    if is_galois() or force:
        show = True if save_path is None else show
        test = convert_synth_offsets_to_gt(offsets)
        return draw_from_gt(test, use_stroke_number=False, show=show, save_path=save_path, linewidth=2, **kwargs)

def test_gt():
    gt = np.load("archidata/all_data_v4.npy", allow_pickle=True)
    item =gt[0]["stroke"]
    print("max", np.max(item[:,0]))

    print(item.shape)
    img = draw_from_gt(item, use_stroke_number=False, show=True)
    offsets = convert_gts_to_synth_format(item)
    print(offsets.shape)
    test = convert_synth_offsets_to_gt(offsets)
    print(test.shape)
    img = draw_from_gt(test, use_stroke_number=False, show=True)
    print("max", np.max(test[:,0]))
    plt.hist(test[:,0])
    plt.hist(offsets[:, 0])

def get_widths():
    gt = np.load("archidata/all_data_v4.npy", allow_pickle=True)
    plt.hist([len(i["stroke"]) for i in gt])
    plt.show()

EPSILON=1e-6
def resample_coords(coords, plot=False):
    """ Should be in EOS format

    """
    # L X (X,Y,EOS)
    coords = coords.astype(np.float64)
    coords2 = np.concatenate([[np.zeros(3)],coords], axis=0)
    #coords2[0,2]=1
    sos = eos_to_sos(coords[:,2])
    not_sos = 1-sos
    distances = np.sum((coords2[1:,:2]-coords2[0:-1, :2])**2, axis=1)**.5*not_sos + EPSILON*sos
    distances[0] = 0
    cum_distance = np.cumsum(distances, axis=0)
    coords[:,:2] -= np.min(coords[:,:2], axis=0)
    factor = 1/np.max(coords[:,1])
    coords[:,:2] *= factor

    start_times = cum_distance[sos==1]

    estimated_width = int(np.max(coords[:,0])*61*1.02/3)
    x_func = interpolate.interp1d(cum_distance, coords[:, 0])
    y_func = interpolate.interp1d(cum_distance, coords[:, 1])
    x,y,sos2 = stroke_recovery.sample(x_func, y_func, start_times, number_of_samples=estimated_width, plot=plot, last_time=cum_distance[-1])
    eos2 = sos_to_eos(sos2)
    final=np.concatenate([[x],[y],[eos2]]).transpose()

    return final

if __name__=="__main__":
    print(project_root())
    print(get_folder("data"))
    print(get_max_checkpoint("./checkpoints/original"))
    get_widths()
    test_gt()