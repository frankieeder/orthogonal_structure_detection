import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import time
from scipy.spatial import Voronoi, voronoi_plot_2d
from mpl_toolkits.mplot3d import Axes3D

root = './s3dis'
subfolders = lambda dir: next(os.walk(dir))[1]
areas = subfolders(root)
areas = [root + '/' + a for a in areas]
areas = [[a + '/' + r + '/' + r + ".txt" for r in subfolders(a)] for a in areas]


pcs = []
area_1_dir = './area_1_pc'
"""if not os.path.isfile(area_1_dir):"""
for area in areas[:1]:
    area_rooms = []
    for room in area[:]:
        print(room)
        room_df = pd.read_csv(
            filepath_or_buffer=room,
            sep=' ',
            names=['x', 'y', 'z', 'r', 'g', 'b']
        )
        area_rooms.append(room_df)
    area_reconstruction = pd.concat(area_rooms)
    pcs.append(area_reconstruction)
pc = pcs[0]

PIXEL_SIZE = 0.5 / 39.37  # 1 inches
THRESHOLD = 25

def plt_grey(img):
    plt.imshow(img, cmap="gray")

def normalize_image(img):
    return img * (255 / img.max())

def find_structures(pc):
    def make_image(pc, a1, a2):
        min_a1, max_a1 = min(pc[a1]), max(pc[a1])
        min_a2, max_a2 = min(pc[a2]), max(pc[a2])
        bins = [(max_a1 - min_a1) // PIXEL_SIZE, (max_a2 - min_a2) // PIXEL_SIZE]
        plot = plt.hist2d(x=pc[a1], y=pc[a2], bins=bins)
        plt.clf()
        plt.close()
        image = plot[0]
        image = normalize_image(image)
        #image = image.astype(np.float32)
        return image, bins

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
def find_perpendicular_structures(pc, a1, a2, structure_title):
    #plot_cloud(pc.sample(frac=0.01))
    start = time.time()
    min_a1, max_a1 = min(pc[a1]), max(pc[a1])
    min_a2, max_a2 = min(pc[a2]), max(pc[a2])
    a1n = a1 + '_pix'
    a2n = a2 + '_pix'
    pc[a1n] = (pc[a1] - min_a1) // PIXEL_SIZE
    pc[a2n] = (pc[a2] - min_a2) // PIXEL_SIZE
    bins = [(max_a1 - min_a1) // PIXEL_SIZE, (max_a2 - min_a2) // PIXEL_SIZE]
    print("Prep time: {0}".format(time.time() - start))
    start = time.time()
    plot = plt.hist2d(x=pc[a1], y=pc[a2], bins=bins)
    print("Image Creation: {0}".format(time.time() - start))
    start = time.time()
    plt.clf()
    plt.close()
    print("Plot closing: {0}".format(time.time() - start))
    start = time.time()
    x_y = plot[0]
    x_y = normalize_image(x_y)
    x_y = x_y.astype(np.uint8)
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
        how='left',
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
pc = find_perpendicular_structures(pc, 'x', 'z', 'floor_x')
pc = find_perpendicular_structures(pc, 'y', 'z', 'floor_y')
pc = find_perpendicular_structures(pc, 'x', 'y', 'wall')

plt_grey(x_y)
