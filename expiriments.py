from orthogonal_structure_detection import *
from scipy.misc import imresize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_by_area():
    return None

def test_by_area():
    return None

def build_checkerboard(w, h):
    re = np.r_[w * [0, 1]]  # even-numbered rows
    ro = np.r_[w * [1, 0]]  # odd-numbered rows
    return np.row_stack(h * (re, ro))

def build_checkerboard_df(w, h, square_size):
    base_check = build_checkerboard(w, h)
    check_array = imresize(base_check, int(square_size) * 100, interp='nearest')
    x = []
    y = []
    for i, row in enumerate(check_array):
        for j, value in enumerate(row):
            if value:
                x.append(i)
                y.append(j)

    return pd.DataFrame({'x': x, 'y': y})

def save_warped_checkboxes(warp_params, w, h, square_size):
    pc = build_checkerboard_df(w, h, square_size)
    for p in warp_params:
        warped = twist_data(pc, 'x', 'y', **p)
        plt.hist2d(warped['x_twist'], warped['y_twist'], bins=[w * square_size, h * square_size], cmap="gray")
        plt.savefig('checkbox_p-{0}_r-{1}.png'.format(p['s'], p['r_perc']))
        plt.clf()
        plt.close()

def test_in(vec, array):
    tru = np.zeros(len(vec)).astype(np.bool)
    for val in array:
        tru = tru | (vec == val)
    return tru

def warp_and_test_room(pc, a1, a2, actual_annotations, warp_params, structure_title):
    results = []
    for i, p in enumerate(warp_params):
        twist_data(pc, a1, a2, **p)
        a1n = a1 + "_twist"
        a2n = a2 + "_twist"
        start = time.time()
        pred_vertical_surfaces, pc = find_perpendicular_structures(pc, a1n, a2n)
        prediction_time = time.time() - start
        actual_vertical_surfaces = test_in(pc.annotation, actual_annotations)
        iou = IOU(actual_vertical_surfaces, pred_vertical_surfaces)

        if False:  # Debugging methods
            all = pixelize_and_plot_pc(pc, a1n, a2n)
            plt.clf()
            plt.close()

            pred_wall = pixelize_and_plot_pc(pc[pc[structure_title]], a1n, a2n)
            plt.clf()
            plt.close()

            actual_wall = pixelize_and_plot_pc(pc[actual_vertical_surfaces], a1n, a2n)
            plt.clf()
            plt.close()

        result = {
            'iou': iou,
            'prediction_time': prediction_time,
            'a1': a1,
            'a2': a2,
            'actual_annotations': actual_annotations,
            'structure_title': structure_title
        }
        result = {**result, **p}
        results.append(result)
    results = pd.DataFrame(results)
    return results







"""Helpful Variables"""
warp_params = [
    {'r_perc': 2, 's': 0},
    {'r_perc': 2, 's': 2},
    {'r_perc': 2, 's': 4},
    {'r_perc': 2, 's': 10},
    {'r_perc': 5, 's': 10},
    {'r_perc': 10, 's': 10},
    {'r_perc': 20, 's': 10}
]

"""Checkbox Warp Params"""
#save_warped_checkboxes(warp_params, 11, 11, 100)

"""Run tests on vertical structures"""
vertical_structure_annotations = ['wall', 'door', 'board', 'column', 'window']

#Test individual rooms
for i in range(6, 6):
    rooms = get_data(
        root,
        room_start=0,
        room_end=3,
        room_step=1,
        area_start=i,
        area_end=i+1
    )
    rooms = rooms[0]
    for room in rooms:
        #try:
        out_dir = "./Area_{0}_Room_{1}_results.csv".format(i + 1, room['room'].min())
        print("Making {0}".format(out_dir))
        results = warp_and_test_room(room, 'x', 'y', vertical_structure_annotations, warp_params, 'vertical')
        results.to_csv(out_dir)
        print("\n\n")
        #except:
            #continue

#Test whole areas
for i in range(0, 6):
    #try:
    out_dir = "./Area_{0}_results.csv".format(i + 1)
    print("Making {0}".format(out_dir))
    data = get_data(root, area_start=i, area_end=i+1)
    data = data[0]
    data = pd.concat(data)
    results = warp_and_test_room(data, 'x', 'y', vertical_structure_annotations, warp_params, 'vertical')
    results.to_csv(out_dir)
    print("\n\n")
    #except:
        #continue



