import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import time
import re
import re
import pickle
from scipy.spatial import Voronoi, voronoi_plot_2d
from mpl_toolkits.mplot3d import Axes3D
from skimage.transform import swirl

root = './s3dis'
subfolders = lambda dir: next(os.walk(dir))[1]
structure_pcs = []

PIXEL_SIZE = 0.5 / 39.37  # 1 inches

def pc_to_df(file):
    return pd.read_csv(
            filepath_or_buffer=file,
            sep=' ',
            names=['x', 'y', 'z', 'r', 'g', 'b'])

def twist_data(pc, a1, a2, r_perc, s):
    print("Twisting Data:::")
    print("Normalizing...")
    a1_mean = pc[a1].mean()
    a2_mean = pc[a2].mean()
    a1_norm = pc[a1] - a1_mean
    a2_norm = pc[a2] - a2_mean
    r = r_perc * np.mean([a1_mean, a2_mean])
    print("Radii...")
    p = np.power(np.power(a1_norm, 2) + np.power(a2_norm, 2), .5)
    print("Thetas...")
    theta = np.arctan(a2_norm / a1_norm)
    del a2_norm
    print("Arctan adjustment...")
    arctan_out_of_range = a1_norm < 0
    del a1_norm
    theta = theta + arctan_out_of_range * np.pi
    del arctan_out_of_range
    r = np.log(2) * r / 5
    print("Theta prime...")
    theta_prime = theta + s * np.exp(-1 * p / r)
    del theta
    print("Remap...")
    pc[a1 + "_twist"] = p * np.cos(theta_prime)
    pc[a2 + "_twist"] = p * np.sin(theta_prime)
    print("Done twisting!")
    return pc



def get_data(root):
    print("Loading Data...")
    area_dfs = []
    for area in subfolders(root)[:1]:
        print(area)
        room_dfs = []
        for room in subfolders(root + '/' + area)[::3]:
            print(room)
            this_room_dir = root + "/" + area + '/' + room
            annotations_dir, subdirs, annotated_files = next(os.walk(this_room_dir + "/Annotations"))
            annotated_files = [f for f in annotated_files if f[-4:] == ".txt"]
            annots_df = []
            for file in annotated_files:
                df = pc_to_df(annotations_dir + "/" + file)
                df['annotation'] = re.sub("_.*", "", file)
                annots_df.append(df)
            room_df = pd.concat(annots_df)
            room_df['room'] = room
            room_dfs.append(room_df)
        print("Joining Area...")
        area_df = pd.concat(room_dfs)
        area_df['area'] = area
        area_dfs.append(area_df)
    pc = pd.concat(area_dfs)
    print("Done getting data!")
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
    return pixelizer_and_plotter(pc[a1], pc[a2])

def pixelizer_and_plotter(v1, v2):
    bins = [(v1.max() - v1.min()) // PIXEL_SIZE, (v2.max() - v2.min()) // PIXEL_SIZE]
    img = plt.hist2d(x=v1, y=v2, bins=bins)[0]
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
    NUM_SAMPLES = 1000
    SLOPE_THRESH = 1

    sorto = x_y.flatten()
    sorto.sort()
    sort_sample = sorto[::len(sorto)//NUM_SAMPLES]
    percentile = (np.argmax(np.gradient(sort_sample) >= SLOPE_THRESH) + 1) / NUM_SAMPLES * 100
    plt.plot(sorto)
    plt.clf()
    thresh = np.percentile(sorto, percentile)
    cv.threshold(
        src=x_y,
        dst=x_y,
        thresh=thresh,
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

def IOU(a, b):
    return sum(a & b) / sum(a | b)




#pc = find_perpendicular_structures(pc, 'x', 'z', 'floor_x')
#pc = find_perpendicular_structures(pc, 'y', 'z', 'floor_y')
#actual_floor = pc.annotation.str.contains('floor')
#actual_ceiling = pc.annotation.str.contains('ceiling')
#actual_structure = actual_wall | actual_floor | actual_ceiling
#pred_structure = pc['floor_x'] | pc['floor_y'] | pc['wall']

LOAD_PICKLES = False
SAVE_PICKLES = False

TEST_NORMAL = False
TEST_TWISTED = True

if TEST_NORMAL:
    NORMAL_PICKLE_DIR = "./data_normal.pickle"
    if os.path.isfile(NORMAL_PICKLE_DIR) and LOAD_PICKLES:
        pickle_in = open(NORMAL_PICKLE_DIR, "rb")
        pc = pickle.load(pickle_in)
    else:
        pc = get_data(root)
        if SAVE_PICKLES:
            pickle_out = open(NORMAL_PICKLE_DIR, "wb")
            pickle.dump(pc, pickle_out)
            pickle_out.close()

    pc = find_perpendicular_structures(pc, 'x', 'y', 'vertical')

    actual_vertical_surfaces = \
        (pc.annotation == 'wall') | \
        (pc.annotation == 'door') | \
        (pc.annotation == 'board') | \
        (pc.annotation == 'column') | \
        (pc.annotation == 'window')

    wall_IOU = IOU(actual_vertical_surfaces, pc['vertical'])
    print(wall_IOU)

    all = pixelize_and_plot_pc(pc, 'x', 'y')
    plt.clf()
    plt.close()

    pred_wall = pixelize_and_plot_pc(pc[pc['vertical']], 'x', 'y')
    plt.clf()
    plt.close()

    actual_wall = pixelize_and_plot_pc(pc[actual_vertical_surfaces], 'x', 'y')
    plt.clf()
    plt.close()


if TEST_TWISTED:
    TWISTED_PICKLE_DIR = "./data_twisted.pickle"
    if os.path.isfile(TWISTED_PICKLE_DIR) and LOAD_PICKLES:
        pickle_in = open(TWISTED_PICKLE_DIR, "rb")
        pc = pickle.load(pickle_in)
    else:
        pc = get_data(root)
        pc = twist_data(pc, 'x', 'y', 5, 3)
        if SAVE_PICKLES:
            pickle_out = open(TWISTED_PICKLE_DIR, "wb")
            pickle.dump(pc, pickle_out)
            pickle_out.close()

    pc = find_perpendicular_structures(pc, 'x_twist', 'y_twist', 'vertical')

    actual_vertical_surfaces = \
        (pc.annotation == 'wall') | \
        (pc.annotation == 'door') | \
        (pc.annotation == 'board') | \
        (pc.annotation == 'column') | \
        (pc.annotation == 'window')

    wall_IOU = IOU(actual_vertical_surfaces, pc['vertical'])
    print(wall_IOU)

    all = pixelize_and_plot_pc(pc, 'x_twist', 'y_twist')
    plt.clf()
    plt.close()

    pred_wall = pixelize_and_plot_pc(pc[pc['vertical']], 'x_twist', 'y_twist')
    plt.clf()
    plt.close()

    actual_wall = pixelize_and_plot_pc(pc[actual_vertical_surfaces], 'x_twist', 'y_twist')
    plt.clf()
    plt.close()

