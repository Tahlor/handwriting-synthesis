import utils
import numpy as np
import stroke_recovery
from scipy import interpolate

def get_stroke_distances(x_online):
    sos = utils.eos_to_sos(x_online)[:,:,-1:]
    not_start_stroke_online = 1-np.tile(sos,2)[:,:-1]
    distances = np.sum((x_online[:,1:,:2]-x_online[:,0:-1, :2])**2*not_start_stroke_online, axis=2)
    return distances


if False:
    m = np.load("../archidata/adapted_dtw_v1.npy", allow_pickle=True)
    for i in m:
        if i["id"]=="a01-000u-01":
            print(i.shape)


if True:
    x_taylor = np.load("/media/data/GitHub/handwriting-synthesis/data/processed/offline_drop/x.npy", allow_pickle=True)
    x_online = np.load("/media/data/GitHub/handwriting-synthesis/data/processed/original/x.npy", allow_pickle=True)

    print(x_taylor.shape)
    print(x_online.shape)
    distances = np.sum((x_online[:, 1:, :2] - x_online[:, 0:-1, :2]) ** 2, axis=2)
    print(np.mean(distances))

    distances = np.sum((x_taylor[:, 1:, :2] - x_taylor[:, 0:-1, :2]) ** 2, axis=2)
    print(np.mean(distances))


    print(np.mean(get_stroke_distances(x_online)))
    print(np.mean(get_stroke_distances(x_taylor)))

    print(np.mean(np.max(x_online[:,:,1], axis=1)))
    print(np.mean(np.max(x_taylor[:,:,1], axis=1)))


# Temp
else:
    gt = np.array(range(27)).reshape(9, 3).astype(np.float64)
    gt[:, 2] = [1, 0, 0, 1, 0, 1, 1, 0, 0]
    gt = utils.sos_to_eos(gt)

    x_online = np.tile(gt, [5,1,1])
    m = get_stroke_distances(x_online)

    utils.resample_coords(gt)
    STOP

def nothing():
    x_online = x_taylor

    EPSILON = .0001
    eos = np.tile(x_online[:, :-1, 2:], 2)
    not_start_stroke_online = 1 - np.tile(x_online[:, :-1, 2:], 2) + eos*EPSILON

    # Resample
    cum_distance = np.c_[np.zeros(not_start_stroke_online.shape[0]), np.cumsum(not_start_stroke_online[:,:,0], axis=1)]

    for i, line in enumerate(x_online):
        print(np.max(line[:, 1]))

        x_func = interpolate.interp1d(cum_distance, line[:, 0])
        y_func = interpolate.interp1d(cum_distance, line[:, 1])

        stroke_recovery.sample(x_func, y_func, sos, number_of_samples=4)
        stop



