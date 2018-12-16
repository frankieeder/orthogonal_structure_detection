import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt

#root = "./results_12-10-18/" # Test old methodology, is worse.
root = "./"
files = next(os.walk(root))[2]
area_result_files = filter(lambda x: re.search(r'Area_._results.csv', x), files)
room_result_files = filter(lambda x: re.search(r'Area_._.*_results.csv', x), files)

area_results = []
for area_file in area_result_files:
    area_result = pd.read_csv(root + area_file, index_col=0)
    area_name = area_file[:6]
    area_result['area'] = area_name
    area_results.append(area_result)
area_results = pd.concat(area_results) if area_results else pd.DataFrame()

room_results = []
for room_file in room_result_files:
    room_result = pd.read_csv(root + room_file, index_col=0)
    area_name = room_file[:6]
    room_name = room_file[12:-12]
    room_result['area'] = area_name
    room_result['room'] = room_name
    room_results.append(room_result)
room_results = pd.concat(room_results) if room_results else pd.DataFrame()


"""IOU ANALYSES"""
room_iou_df = room_results\
    .groupby(['r_perc', 's'], as_index=False)\
    .mean()
out_dir = root + "results_iou_data_by_room.csv"
room_iou_df.to_csv(out_dir)
print("Saved individual iou results to {0}\n".format(out_dir))

"""AREA ANALYSIS"""
#FINISH CLEANING THIS UP
area_results = area_results
area_iou_df = area_results\
    .groupby(['r_perc', 's'], as_index=False)\
    .mean()
unmodified_areas = np.array((area_results['s'] == 0))
unmodified_area_ious = area_results[unmodified_areas][['area', 'iou']]
unmodified_area_iou = unmodified_area_ious.iou.mean()
modified_area_ious = area_results[~unmodified_areas][['area', 'iou']]
modified_area_iou = modified_area_ious.groupby(['area'], as_index=False).mean()
modified_area_iou['iou'].mean()
area_aggregated_results.to_csv('area_aggregated_results.csv')
