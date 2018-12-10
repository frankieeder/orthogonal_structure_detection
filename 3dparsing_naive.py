import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import time
import re
from scipy.spatial import Voronoi, voronoi_plot_2d
from mpl_toolkits.mplot3d import Axes3D

root = './s3dis'
subfolders = lambda dir: next(os.walk(dir))[1]
structure_pcs = []

PIXEL_SIZE = 0.5 / 39.37  # 1 inches
THRESHOLD = 25

def pc_to_df(file):
    return pd.read_csv(
            filepath_or_buffer=file,
            sep=' ',
            names=['x', 'y', 'z', 'r', 'g', 'b'])

def get_data(root):
    print("Loading Data...")
    area_dfs = []
    for area in subfolders(root)[:1]:
        room_dfs = []
        for room in subfolders(root + '/' + area)[:1]:
            print(room)
            this_room_dir = root + "/" + area + '/' + room
            annotations_dir, subdirs, annotated_files = next(os.walk(this_room_dir + "/Annotations"))
            annots_df = []
            for file in annotated_files:
                df = pc_to_df(annotations_dir + "/" + file)
                df['annotation'] = file[:-4]
                annots_df.append(df)
            room_df = pd.concat(annots_df)
            room_df['room'] = room
            room_dfs.append(room_df)
        area_df = pd.concat(room_dfs)
        area_df['area'] = area
        area_dfs.append(area_df)
    pc = pd.concat(area_dfs)
    return pc

def plt_grey(img):
    plt.imshow(img, cmap="gray")

def normalize_image(img):
    return img * (255 / img.max())

def rgb_to_hex(r, g, b):
    return list(zip(r.astype(int), g.astype(int), b.astype(int)))

def plot_cloud(pc):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = rgb_to_hex(pc['r'], pc['g'], pc['b'])
    ax.scatter(
        pc['x'],
        pc['y'],
        pc['z'],
        s=0.5
    )

def pixelize_and_plot_pc(pc, a1, a2):
    min_a1, max_a1 = pc[a1].min(), pc[a1].max()
    min_a2, max_a2 = pc[a2].min(), pc[a2].max()
    bins = [(max_a1 - min_a1) // PIXEL_SIZE, (max_a2 - min_a2) // PIXEL_SIZE]
    img = plt.hist2d(x=pc[a1], y=pc[a2], bins=bins)[0]
    return img.T.copy()

def find_perpendicular_structures(pc, a1, a2, structure_title, merge_type='left'):
    #plot_cloud(pc.sample(frac=0.01))
    start = time.time()
    a1n = a1 + '_pix'
    a2n = a2 + '_pix'
    pc[a1n] = (pc[a1] - pc[a1].min()) // PIXEL_SIZE
    pc[a2n] = (pc[a2] - pc[a2].min()) // PIXEL_SIZE
    print("Prep time: {0}".format(time.time() - start))
    start = time.time()
    x_y = pixelize_and_plot_pc(pc, a1, a2)
    print("Image Creation: {0}".format(time.time() - start))
    plt.clf()
    plt.close()
    start = time.time()
    x_y = normalize_image(x_y)
    x_y = x_y.astype(np.uint8)
    #plt_grey(x_y)
    print("Image extraction and normalization: {0}".format(time.time() - start))
    start = time.time()
    #image = image.astype(np.float32)

    #x_y = make_image(pc, 'x', 'y')
    #plt_grey(x_y_proj)

    #blur = cv.GaussianBlur(x_y, (3, 3), 0)
    #thresh, x_y_structs = cv.threshold(blur, 0, 255, cv.THRESH_BINARY)
    thresh, x_y_structs = cv.threshold(
        src=x_y,
        dst=x_y,
        thresh=THRESHOLD,
        maxval=255,
        type=cv.THRESH_BINARY
    )
    print("Thresholding: {0}".format(time.time() - start))
    start = time.time()
    #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    #x_y = cv.dilate(x_y, kernel, iterations=1)
    #x_y = cv.morphologyEx(x_y, cv.MORPH_OPEN, kernel)


    #plt_grey(x_y)

    wall_points = np.squeeze(cv.findNonZero(x_y))
    print("Find structure points: {0}".format(time.time() - start))
    start = time.time()
    wall_df = pd.DataFrame(
        wall_points,
        columns=[a1n, a2n]
    )
    wall_df[structure_title] = True
    print("Form structure df: {0}".format(time.time() - start))
    start = time.time()

    merged = pc.merge(
        right=wall_df,
        how=merge_type,
        on=[a1n, a2n],
        copy=False
    )
    print("Merge to original pc: {0}".format(time.time() - start))
    start = time.time()
    merged = merged.drop([a1n, a2n], axis=1)
    pc = pc.drop([a1n, a2n], axis=1)
    merged[structure_title] = merged[structure_title].fillna(False)
    print("Final pc cleanup: {0}".format(time.time() - start))
    return merged
    #vor = Voronoi(wall_points)
    #voronoi_plot_2d(vor)

def IOU(a, b):
    return sum(a & b) / sum(a | b)

pc = get_data(root)
#pc = find_perpendicular_structures(pc, 'x', 'z', 'floor_x')
#pc = find_perpendicular_structures(pc, 'y', 'z', 'floor_y')
pc = find_perpendicular_structures(pc, 'x', 'y', 'wall')
actual_wall = pc.annotation.str.contains('wall')
actual_door = pc.annotation.str.contains('door')
actual_vertical_surfaces = actual_wall | actual_door
#actual_floor = pc.annotation.str.contains('floor')
#actual_ceiling = pc.annotation.str.contains('ceiling')
#actual_structure = actual_wall | actual_floor | actual_ceiling
#pred_structure = pc['floor_x'] | pc['floor_y'] | pc['wall']

all = pixelize_and_plot_pc(pc, 'x', 'y')
plt.clf()
plt.close()

actual_wall = pixelize_and_plot_pc(pc[actual_wall], 'x', 'y')
plt.clf()
plt.close()

wall_IOU = IOU(actual_vertical_surfaces, pc['wall'])
print(wall_IOU)
plt_grey(x_y)
