import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import time
import re

root = './s3dis'
subfolders = lambda dir: next(os.walk(dir))[1]

PIXEL_SIZE = 0.5 / 39.37  # .5 inches

def pc_to_df(file):
    """
    Simple helper function to help read S3DIS dataset.
    :param file:
    :return:
    """
    return pd.read_csv(
            filepath_or_buffer=file,
            sep=' ',
            names=['x', 'y', 'z', 'r', 'g', 'b'])

def twist_data(pc, a1, a2, s):
    """
    Note: a1 and a2 denote the plane along which to twist, orthogonal to the axis about which we are twisting.
    :param pc: input point cloud, as pandas dataframe.
    :param a1: first axis along which to twist, as string denoting column name.
    :param a2: second axis along which to twist, as string denoting column name.
    :param s: the strength of the twist.
    :return: twisted point cloud, as pandas dataframe.

    """
    if s == 0:
        pc[a1 + "_twist"] = pc[a1]
        pc[a2 + "_twist"] = pc[a2]
        return pc
    print("Twisting Data:::")
    print("Normalizing...")
    a1_mean = pc[a1].mean()
    a2_mean = pc[a2].mean()
    a1_norm = pc[a1] - a1_mean
    a2_norm = pc[a2] - a2_mean
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
    print("Theta prime...")
    normalized_p = p / p.max() #Normalize radii to [0, 1]
    theta_prime = theta + s * normalized_p
    del theta
    print("Remap...")
    pc[a1 + "_twist"] = p * np.cos(theta_prime)
    pc[a2 + "_twist"] = p * np.sin(theta_prime)
    print("Done twisting!")
    return pc



def get_data(root,
             area_start=0, area_end=None, area_step=1,
             room_start=0, room_end=None, room_step=1):
    """
    Helper function to help load point cloud data from the S3DIS dataset.
    :param root: The location of the s3dis dataset.
    :param area_start: The first room to collect.
    :param area_end: The last room to collect.
    :param area_step: The step size for our iteration between area_start and area_end
    :param room_start: The first room in each area that we want to collect.
    :param room_end: The last room in each area that we want to collect.
    :param room_step: Similar to area_step, the step size for our room collection iteration.
    :return: A list of lists of dataframes that correspond to the queried data.
    """
    print("Loading Data...")
    area_dfs = []
    areas = subfolders(root)
    if area_end is None:
        area_end = len(areas)
    for area in areas[area_start:area_end:area_step]:
        print(area)
        room_dfs = []
        rooms = subfolders(root + '/' + area)
        if room_end is None:
            room_end = len(rooms)
        for room in rooms[room_start:room_end:room_step]:
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
        for room_df in room_dfs:
            room_df['area'] = area
        area_dfs.append(room_dfs)
    print("Done getting data!")
    return area_dfs

def pixelize_and_plot_pc(pc, a1, a2):
    """Helper function to plot histogram density map for debugging."""
    return pixelizer_and_plotter(pc[a1], pc[a2])

def pixelizer_and_plotter(v1, v2):
    """Plots histogram density map for debugging."""
    bins = [(v1.max() - v1.min()) // PIXEL_SIZE, (v2.max() - v2.min()) // PIXEL_SIZE]
    img = plt.hist2d(x=v1, y=v2, bins=bins)[0]
    return img.T.copy()

def find_perpendicular_structures(pc, a1, a2):
    """Our main algorithm. Note that we add inputs a1 and a2 to generalize to projections in the xz and yz planes,
    but for the purposes of the current paper we always project to the xy-plane to detect walls.
    :param pc: input point cloud in which to detect structures
    :param a1: first axis denoting plane to project to, as string.
    :param a2: second axis denoting plane to project to, as string.
    :return: A reordered version of the input point cloud, along with a vector of indices that denote points
        detected as structures.
    """
    # Define helpful constants
    PAD_WIDTH = 10

    def crop(img):
        return img[PAD_WIDTH:-PAD_WIDTH, PAD_WIDTH:-PAD_WIDTH]

    def pad(img):
        return np.pad(img, PAD_WIDTH, mode='constant', constant_values=1)

    # Define helpful variables.
    start = time.time()
    a1n = a1 + '_pix'
    a2n = a2 + '_pix'
    pc[a1n] = ((pc[a1] - pc[a1].min()) // PIXEL_SIZE)
    pc[a2n] = ((pc[a2] - pc[a2].min()) // PIXEL_SIZE)
    v1 = pc[a1n]
    v2 = pc[a2n]
    bins = [int((pc[a1].max() - pc[a1].min()) // PIXEL_SIZE), int((pc[a2].max() - pc[a2].min()) // PIXEL_SIZE)]
    print("Prep time: {0}".format(time.time() - start))

    #Create density map.
    start = time.time()
    hist, x_edges, y_edges = np.histogram2d(v1, v2, bins)
    hist = hist.T.copy()
    print("Histogram creation: {0}".format(time.time() - start))

    # Find void space, open, dilate, and isolate.
    start = time.time()
    void = (hist == 0).astype(np.uint8)
    void = pad(void)
    hist = pad(hist)
    kernel = np.ones((3,3), np.uint8)
    opened = cv.morphologyEx(void, cv.MORPH_OPEN, kernel)
    dilated = cv.dilate(opened, kernel, iterations=3)
    wall = dilated - opened
    print("Opening and Dilation: {0}".format(time.time() - start))

    # Save plot visuals for Figure 1
    if False:
        fig = plt.figure(frameon=False)
        #fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(hist)
        fig.savefig('visuals_hist.png', dpi=300)
        ax.imshow(img)
        fig.savefig('visuals_void_space.png', dpi=300)
        ax.imshow(opened)
        fig.savefig('visuals_opened.png', dpi=300)
        ax.imshow(dilated)
        fig.savefig('visuals_dilated.png', dpi=300)
        ax.imshow(wall)
        fig.savefig('visuals_wall.png', dpi=300)
        fig.clf()
    # Make sure to crop walls back for remapping step!
    wall = crop(wall)
    hist = crop(hist)

    # Structure our known wall points
    start = time.time()
    wall_points = np.squeeze(cv.findNonZero(wall))
    print("Find structure points: {0}".format(time.time() - start))
    start = time.time()
    wall_df = pd.DataFrame(
        wall_points,
        columns=[a1n, a2n]
    )
    wall_df['orthog'] = True
    print("Form structure df: {0}".format(time.time() - start))

    # Remap our known wall points to our original point cloud
    start = time.time()
    merged = pc.merge(
        right=wall_df,
        how='left',
        on=[a1n, a2n],
        copy=False
    )
    print("Merge to original pc: {0}".format(time.time() - start))

    #Create our desired output as a vector of indices of the merged dataframe that are wall points
    start = time.time()
    result = merged["orthog"].fillna(False)
    print("Results Casting: {0}".format(time.time() - start))

    # Save plot visuals for Figure 2
    if False:
        fig = plt.figure(frameon=False)
        # fig.set_size_inches(w, h)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        test_pixels = hist.copy()
        test_pixels = np.maximum(test_pixels, wall*test_pixels.max())
        test_pixels = pad(test_pixels)

        ax.imshow(test_pixels)
        fig.savefig('visuals_wall_and_hist.png', dpi=300)

    #Drop redundant data from original point cloud to leave it unmodified.
    start = time.time()
    merged.drop(["orthog"], axis=1, inplace=True)
    print("Final pc cleanup: {0}".format(time.time() - start))

    return result, merged

def IOU(a, b):
    return sum(a & b) / sum(a | b)
