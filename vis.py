import numpy as np
from open3d import *
import pandas as pd
import os
import pickle
import sys
import matplotlib.pyplot as plt

def save_as_pickled_object(obj, filepath):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


def try_to_load_as_pickled_object_or_None(filepath):
    """
    This is a defensive way to write pickle.load, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    try:
        input_size = os.path.getsize(filepath)
        bytes_in = bytearray(0)
        with open(filepath, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        obj = pickle.loads(bytes_in)
    except:
        return None
    return obj

root = './s3dis'
subfolders = lambda dir: next(os.walk(dir))[1]
areas = subfolders(root)
areas = [root + '/' + a for a in areas]
areas = [[a + '/' + r + '/' + r + ".txt" for r in subfolders(a)] for a in areas]


pcs = []
area_1_dir = './area_1_pc'
if not os.path.isfile(area_1_dir):
    for area in areas[:1]:
        area_rooms = []
        for room in area:
            print(room)
            room_df = pd.read_csv(
                filepath_or_buffer=room,
                sep=' ',
                names=['x', 'y', 'z', 'r', 'g', 'b']
            )
            area_rooms.append(room_df)
        area_reconstruction = pd.concat(area_rooms)
        pcs.append(area_reconstruction)
if not pcs:
    area_1 = try_to_load_as_pickled_object_or_None(area_1_dir)
else:
    area_1 = pcs[0]
    save_as_pickled_object(area_1, area_1_dir)

if __name__ == "__main__":
    print("Load a ply point cloud, print it, and render it")
    pcd = read_point_cloud("./s3Dis/Area_1_conference_Room_1/conferenceRoom_1.txt")
    print(pcd)
    print(np.asarray(pcd.points))
    draw_geometries([pcd])

    print("Downsample the point cloud with a voxel of 0.05")
    downpcd = voxel_down_sample(pcd, voxel_size = 0.05)
    draw_geometries([downpcd])

    print("Recompute the normal of the downsampled point cloud")
    estimate_normals(downpcd, search_param = KDTreeSearchParamHybrid(
            radius = 0.1, max_nn = 30))
    draw_geometries([downpcd])

    print("Print a normal vector of the 0th point")
    print(downpcd.normals[0])
    print("Print the normal vectors of the first 10 points")
    print(np.asarray(downpcd.normals)[:10,:])
    print("")

    print("Load a polygon volume and use it to crop the original point cloud")
    vol = read_selection_polygon_volume("../../TestData/Crop/cropped.json")
    chair = vol.crop_point_cloud(pcd)
    draw_geometries([chair])
    print("")

    print("Paint chair")
    chair.paint_uniform_color([1, 0.706, 0])
    draw_geometries([chair])
    print("")