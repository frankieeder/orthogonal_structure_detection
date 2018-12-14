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
unmodified_rooms = room_results['s'] == 0
unmodified_room_ious = room_results[unmodified_rooms][['area', 'room', 'iou']]
unmodified_room_iou = unmodified_room_ious.iou.mean()
modified_room_ious = room_results[~unmodified_rooms][['area', 'room', 'iou']]
modified_room_median_ious = modified_room_ious\
    .groupby(['area', 'room'], as_index=False)\
    .median()
room_iou_df = unmodified_room_ious.merge(
    modified_room_median_ious,
    on=['area', 'room'],
    suffixes=['_unmodified', '_modified']
)
room_iou_df['del_iou'] = room_iou_df.iou_modified - room_iou_df.iou_unmodified

# Output
out_dir = root + "results_iou_data_by_room.csv"
iou_unmodified_mean = room_iou_df.iou_unmodified.mean()
iou_modified_mean = room_iou_df.iou_modified.mean()
del_iou_mean = room_iou_df.del_iou.mean()
room_iou_df.to_csv(out_dir)
print("Mean IOU of unmodified rooms = {0}.".format(iou_unmodified_mean))
print("Mean IOU of modified rooms = {0}.".format(iou_modified_mean))
print("Mean ∆IOU = {0}.".format(del_iou_mean))
print("Saved individual iou results to {0}\n".format(out_dir))

"""PERFORMANCE ANALYSES"""
room_times = room_results[['area', 'room', 'prediction_time']]\
    .groupby(['area', 'room'], as_index=False)\
    .mean()

# Output
out_dir = root + "results_performance_data_by_room.csv"
room_time_mean = room_times.prediction_time.mean()
room_times.to_csv(out_dir)
print("Mean execution time of rooms = {0}.".format(room_time_mean))
print("Saved individual performance results to {0}\n".format(out_dir))

"""AREA ANALYSIS"""
#FINISH CLEANING THIS UP
area_results = area_results.reset_index()
unmodified_areas = np.array((area_results['s'] == 0))
unmodified_area_ious = area_results[unmodified_areas][['area', 'iou']]
unmodified_area_iou = unmodified_area_ious.iou.mean()
modified_area_ious = area_results[~unmodified_areas][['area', 'iou']]
modified_area_iou = modified_area_ious.groupby(['area'], as_index=False).mean()
modified_area_iou['iou'].mean()
area_aggregated_results.to_csv('area_aggregated_results.csv')
