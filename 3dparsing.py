import pandas as pd
import scipy.io
import h5py
pc_mat_dir = './2d3ds/area_3/3d/pointcloud.mat'
"""with h5py.File(pc_dir, 'r') as f:
    f.keys()"""

pc = pd.read_csv(
    filepath_or_buffer="./s3dis/Area_2/WC_1/WC_1.txt",
    sep=' ',
    names=['x', 'y', 'z', 'r', 'g', 'b']
)

min_x = min(pc['x'])
min_y = min(pc['y'])
min_z = min(pc['z'])
pixel_size = 1 / 39.37 #1 inches
pc['x_n'] = (pc['x'] - min_x) // pixel_size
pc['y_n'] = (pc['y'] - min_y) // pixel_size
pc['z_n'] = (pc['z'] - min_z) // pixel_size
counts = pc.groupby(['x_n', 'y_n', 'z_n'], as_index=False).size()



