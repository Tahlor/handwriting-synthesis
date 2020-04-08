import warnings
import numpy as np
import stroke_recovery
import multiprocessing

PARAMETER = "d"

def resample_one(self, item, parameter=PARAMETER):
    """ Resample will be based on time, unless the number of samples has been calculated;
            this is only calculated if you supply a pickle file or a CNN! In this case the number
            of stroke points corresponds to the image width in pixels. Otherwise:
                * The number of stroke points corresponds to how long it took to write
                OR
                * If "scale_time_distance" was selected. the number of stroke points corresponds to how long
                the strokes are
    Args:
        item: Dictionary with a "raw" dictionary item
    Returns:
        Adds/modifies the "gt" key

    """
    output = stroke_recovery.prep_stroke_dict(item["raw"])  # returns dict with {x,y,t,start_times,x_to_y (aspect ratio), start_strokes (binary list), raw strokes}
    x_func, y_func = stroke_recovery.create_functions_from_strokes(output,
                                                                   parameter=parameter)  # can be d if the function should be a function of distance
    if "number_of_samples" not in item:
        item["number_of_samples"] = int(output[parameter + "range"] / self.interval)
        warnings.warn("UNK NUMBER OF SAMPLES!!!")

    if parameter == "t":
        start_times = output.start_times
    elif parameter == "d":
        start_times = output.start_distances
        item["start_distances"] = output.start_distances
    else:
        raise NotImplemented(f"Unknown {parameter}")

    gt = create_gts(x_func, y_func, start_times=start_times,
                    number_of_samples=item["number_of_samples"],
                    noise=self.noise,
                    gt_format=self.gt_format)

    item["gt"] = gt  # {"x":x, "y":y, "is_start_time":is_start_stroke, "full": gt}
    item["x_func"] = x_func
    item["y_func"] = y_func
    return item


def resample_data(self, data_list, parallel=True):
    if parallel and False:
        poolcount = max(1, multiprocessing.cpu_count() - 3)
        pool = multiprocessing.Pool(processes=poolcount)
        all_results = list(
            pool.imap_unordered(self.resample_one, tqdm(data_list)))  # iterates through everything all at once
        pool.close()
    else:
        all_results = []
        for item in data_list:
            all_results.append(self.resample_one(item))
    return all_results

def create_gts(x_func, y_func, start_times, number_of_samples, gt_format, noise=None):
    """ Return LENGTH X VOCAB

    Args:
        x_func:
        y_func:
        start_times: [.1,1.2,3.6,...]
        number_of_samples: Number of GT points
        noise: Add some noise to the sampling
        start_of_stroke_method: "normal" - use 1's for starts
                              "interpolated" - use 0's for starts, 1's for ends, and interpolate
                              "interpolated_distance" - use 0's for starts, total distance for ends, and interpolate

    Returns:
        gt array: SEQ_LEN x [X, Y, IS_STROKE_START, IS_END_OF_SEQUENCE]
    """
    # Sample from x/y functions
    x, y, is_start_stroke = stroke_recovery.sample(x_func, y_func, start_times,
                                                   number_of_samples=number_of_samples, noise=noise)

    # Put it together
    gt = []

    for i,el in enumerate(gt_format):
        if el == "x":
            gt.append(x)
        elif el == "y":
            gt.append(y)
        elif el == "sos":
            gt.append(is_start_stroke)
        elif el == "eos":
            # Create GT matrix
            end_of_sequence_flag = np.zeros(x.shape[0])
            end_of_sequence_flag[-1] = 1

            gt.append(end_of_sequence_flag)
        elif "sos_interp" in el:
            # Instead of using 1 to denote start of stroke, use 0, increment for each additional stroke based on distance of stroke
            is_start_stroke = stroke_recovery.get_stroke_length_gt(x, y, is_start_stroke, use_distance=(el=="sos_interp_dist"))
            gt.append(is_start_stroke)
        elif el == "stroke_number": # i.e. 1,1,1,1,1,2,2,2,2,2...
            stroke_number = np.cumsum(is_start_stroke)
            gt.append(stroke_number)
        else:
            raise Exception(f"Unkown GT format: {el}")

    gt = np.array(gt).transpose([1,0]) # swap axes -> WIDTH, VOCAB
    return gt
