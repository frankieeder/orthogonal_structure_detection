import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import scipy.io
#import h5py
pc_mat_dir = './2d3ds/area_3/3d/pointcloud.mat'
"""with h5py.File(pc_dir, 'r') as f:
    f.keys()"""

pc = pd.read_csv(
    filepath_or_buffer="./s3dis/Area_2/WC_1/WC_1.txt",
    sep=' ',
    names=['x', 'y', 'z', 'r', 'g', 'b']
)

PIXEL_SIZE = 1 / 39.37  # 1 inches

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

def make_image(df):
    x_label, y_label, v_label = df.columns
    x_size = int(max(df[x_label])) + 1
    y_size = int(max(df[y_label])) + 1
    image = np.zeros((x_size, y_size))
    for r, data in df.iterrows():
        x, y, v = data.tolist()
        image[x][y] = v
    return image


pixelize_and_normalize(pc)
find_walls(pc)