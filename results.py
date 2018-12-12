import pandas as pd
import os
import re

files = next(os.walk("./"))[2]
area_result_files = filter(lambda x: re.search(r'Area_._results.csv', x), files)
room_result_files = filter(lambda x: re.search(r'Area_._.*_results.csv', x), files)

area_results = []
for area_file in area_result_files:
    area_result = pd.read_csv(area_file, index_col=0)
    area_name = area_file[:6]
    area_result['area'] = area_name
    area_results.append(area_result)
area_results = pd.concat(area_results)

room_results = []
for room_file in room_result_files:
    room_result = pd.read_csv(room_file, index_col=0)
    area_name = room_file[:6]
    room_name = room_file[12:-12]
    room_result['area'] = area_name
    room_result['room'] = room_name
    room_results.append(room_result)
room_results = pd.concat(room_results)

room_aggregated_results = room_results.groupby(['area', 'r_perc', 's'], as_index=False).mean()
room_aggregated_results.to_csv('room_aggregated_results.csv')
area_aggregated_results = area_results.groupby(['r_perc', 's'], as_index=False).mean()
area_aggregated_results.to_csv('area_aggregated_results.csv')
