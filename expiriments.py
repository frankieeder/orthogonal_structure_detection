from orthogonal_structure_detection import *
import pandas as pd
import numpy as np

def test_in(vec, array):
    """
    Simple function to test if each element of input vector is in the input array
    :param vec: input numpy array to test for membership
    :param array: the set of possible values for membership
    :return: vector with trues if element in array false otherwise.
    """
    tru = np.zeros(len(vec)).astype(np.bool)
    for val in array:
        tru = tru | (vec == val)
    return tru

def warp_and_test_room(pc, a1, a2, actual_annotations, warp_params):
    """
    Helper functions to test different parameters and write results
    :param pc: point cloud on which to test
    :param a1: first axis denoting plane to project to, as string.
    :param a2: second axis denoting plane to project to, as string.
    :param actual_annotations: The annotations from s3dis to define as walls for our purposes
    :param warp_params: A list of parameters to pass to data twisting.
    :return: results as a dataframe.
    """
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

        result = {
            'iou': iou,
            'prediction_time': prediction_time,
            'a1': a1,
            'a2': a2,
            'actual_annotations': actual_annotations
        }
        result = {**result, **p}
        results.append(result)
    results = pd.DataFrame(results)
    return results







"""Helpful Variables"""
warp_params = [
    {'s': 0},
    {'s': np.pi/4},
    {'s': 2 * np.pi},
]

"""Run tests on vertical structures"""
vertical_structure_annotations = ['wall', 'board', 'column', 'window'] #Defines our wall class in comparison to Armeni


#Test individual rooms
for i in range(0, 6):
    rooms = get_data(
        root,
        area_start=i,
        area_end=i+1
    )
    rooms = rooms[0]
    for room in rooms:
        out_dir = "./Area_{0}_Room_{1}_results.csv".format(i + 1, room['room'].min())
        print("Making {0}".format(out_dir))
        results = warp_and_test_room(room, 'x', 'y', vertical_structure_annotations, warp_params)
        results.to_csv(out_dir)
        print("Success! Saved csv...\n\n")

#Test whole areas
for i in range(0, 6):
    out_dir = "./Area_{0}_results.csv".format(i + 1)
    print("Making {0}".format(out_dir))
    data = get_data(
        root,
        area_start=i,
        area_end=i+1
    )
    data = data[0]
    data = pd.concat(data)
    results = warp_and_test_room(data, 'x', 'y', vertical_structure_annotations, warp_params, 'vertical')
    results.to_csv(out_dir)
    print("Success! Saved csv...\n\n")



