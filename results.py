import pandas as pd
import os
import re

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
out_dir = root + "results_by_room.csv"
print("Room IOU results:", room_iou_df)
room_iou_df.to_csv(out_dir)
print("Saved room iou results to {0}\n".format(out_dir))

"""AREA ANALYSIS"""
area_results = area_results
area_iou_df = area_results\
    .groupby(['r_perc', 's'], as_index=False)\
    .mean()
print("Area IOU results:", area_iou_df)
out_dir = root + "results_by_area.csv"
room_iou_df.to_csv(out_dir)
print("Saved area iou results to {0}\n".format(out_dir))
