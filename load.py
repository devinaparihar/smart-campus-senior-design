import os

import numpy as np

def load_from_dir(data_root):
    month_list = [f for f in os.listdir(data_root) if not os.path.isfile(os.path.join(data_root, f))]
    data = {}
    for m in month_list:
        data[str(m)] = {}
        m_path = os.path.join(data_root, m)
        day_list = [f for f in os.listdir(m_path) if not os.path.isfile(os.path.join(m_path, f))]
        for d in day_list:
            data[str(m)][str(d)] = {}
            d_path = os.path.join(m_path, d)
            individual_list = [f for f in os.listdir(d_path) if os.path.isfile(os.path.join(d_path, f))]
            for i in individual_list:
                i_data = np.load(os.path.join(d_path, i))
                # Only save unix time
                data[str(m)][str(d)][str(i)] = i_data[:, :3]
    return data

def compile_data_by_individual(month_data):
    individual_data = {}
    for d in range(1, 31):
        d_str = "{:02d}".format(d)
        if d_str not in month_data.keys():
            break
        day_data = month_data[d_str]
        for i in day_data.keys():
            if i not in individual_data.keys():
                individual_data[i] = day_data[i]
            else:
                individual_data[i] = np.concatenate((individual_data[i], day_data[i]))
    return individual_data

def load_cleaned_individuals(data_root):
    data = []
    individual_list = [f for f in os.listdir(data_root) if os.path.isfile(os.path.join(data_root, f))]
    for f in individual_list:
        data.append(np.load(os.path.join(data_root, f)))
    return data
