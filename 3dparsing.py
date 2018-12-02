import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
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
    for room in area[:1]:
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
    return image

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

#plot_cloud(pc.sample(frac=0.01))

x_y = make_image(pc, 'x', 'y')
#plt_grey(x_y_proj)
x_y = x_y.astype(np.uint8)
cv.threshold(x_y, THRESHOLD, 255, 0, x_y)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
#x_y = cv.dilate(x_y, kernel, iterations=1)
#x_y = cv.morphologyEx(x_y, cv.MORPH_OPEN, kernel)


plt_grey(x_y)

wall_points = np.squeeze(cv.findNonZero(x_y))
#vor = Voronoi(wall_points)
#voronoi_plot_2d(vor)

plt_grey(x_y)
x_y_gradient = cv.Laplacian(x_y_proj, cv.CV_32F)
x_y_gradient = abs(x_y_gradient)
#plt_grey(x_y_gradient)
corners = cv.cornerHarris(x_y_gradient, 2, 3, 0.04)
corners[corners < 0] = 0
corners = normalize_image(corners)
corners[corners > 0.001 * corners.max()] = 255
corners[corners < 0] = 255
plt_grey(corners)
x_z_proj = make_image(pc, 'x', 'z')
y_z_proj = make_image(pc, 'y', 'z')
x = 2

# BAD BELOW HERE
def pixelize_and_normalize(pc, pixel_size=PIXEL_SIZE):
    min_x = min(pc['x'])
    min_y = min(pc['y'])
    min_z = min(pc['z'])
    pc['x_n'] = ((pc['x'] - min_x) / pixel_size).astype(int)
    pc['y_n'] = ((pc['y'] - min_y) / pixel_size).astype(int)
    pc['z_n'] = ((pc['z'] - min_z) / pixel_size).astype(int)

def find_walls(pc):
    x_y = group_as_df(pc, ['x_n', 'y_n'])
    x_y_proj = make_image(x_y)
    imgplot = plt.imshow(x_y_proj)
    x_z = pc.groupby(['x_n', 'z_n'], as_index=False).size()
    x_z_proj = make_image(x_z)
    y_z = pc.groupby(['y_n', 'z_n'], as_index=False).size()
    x_z_proj = make_image(y_z)

def group_as_df(pc, by):
    return pd.DataFrame({'value' : pc.groupby(by).size()}).reset_index()

def make_image_old(df):
    x_label, y_label, v_label = df.columns
    x_size = int(max(df[x_label])) + 1
    y_size = int(max(df[y_label])) + 1
    image = np.zeros((x_size, y_size))
    for r, data in df.iterrows():
        x, y, v = data.tolist()
        image[x][y] = v
    return image


#pixelize_and_normalize(pc)
#find_walls(pc)