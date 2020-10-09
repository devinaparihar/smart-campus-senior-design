import os
import sys
import csv

import numpy as np

"""
Handle exception that os.mkdir throws when directory already exists
"""
def makedir(path):
    try:
        os.mkdir(path)
    except:
        print(str(path) + ' directory may already exist')

"""
Given a file (csv) creates a different numpy matrix for each publisher_id
containing time, latitude, longitude, and altitude for each data point
sorted in chronological order (file saved to output directory)
"""
def extract_from_file(file_path, output_dir):
    individual_data = {}
    with open(os.path.join(file_path)) as csvfile:
        reader = csv.reader(csvfile)
        column_to_index = None
        for i, row in enumerate(reader):
            if column_to_index is None:
                column_to_index = {}
                for i, column_label in enumerate(row):
                    column_to_index[column_label] = i
            else:
                key = row[column_to_index['publisher_id']]
                data_point = []
                data_point.append(int(row[column_to_index['location_at']]))
                data_point.append(float(row[column_to_index['latitude']]))
                data_point.append(float(row[column_to_index['longitude']]))
                try:
                    data_point.append(float(row[column_to_index['altitude']]))
                except:
                    data_point.append(0)
                if key not in individual_data.keys():
                    individual_data[key] = []
                individual_data[key].append(data_point)

    for key in individual_data.keys():
        mat = np.array(individual_data[key])
        mat = mat[mat[:, 0].argsort()]
        np.save(os.path.join(output_dir, str(key)), mat)

"""
Do the extraction for each month and each day in every month where the
files for each month and day are stored like this in the root directory:
-root
    -month1
        -day1.csv
        ...
    -month2
        -day2.csv
        ...
    ...
"""
def traverse_files(raw_data_root, individual_data_root):
    directory_list = [f for f in os.listdir(raw_data_root) if not os.path.isfile(os.path.join(raw_data_root, f))]
    for d in directory_list:
        raw_d_path = os.path.join(raw_data_root, d)
        individual_d_path = os.path.join(individual_data_root, d)
        makedir(individual_d_path)
        file_list = [f for f in os.listdir(raw_d_path) if os.path.isfile(os.path.join(raw_d_path, f))]
        for f in file_list:
            individual_f_dir_path = os.path.join(individual_d_path, os.path.splitext(f)[0])
            makedir(individual_f_dir_path)
            extract_from_file(os.path.join(raw_d_path, f), individual_f_dir_path)

"""
First argument is raw data root directory and second is output, individual data root directory
Sample:
python extract.py ./raw_data/ ./individual_data/
"""
if __name__ == '__main__':
    # Make csv size limit as big as possible on machine
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = max_int // 10

    traverse_files(sys.argv[1], sys.argv[2])
