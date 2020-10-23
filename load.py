import os

import numpy as np

def load_from_dir(data_root):
    month_list = [f for f in os.listdir(data_root) if not os.path.isfile(os.path.join(data_root, f))]
    data = {}
    for m in month_list:
        data[m] = {}
        m_path = os.path.join(data_root, m)
        day_list = [f for f in os.listdir(m_path) if not os.path.isfile(os.path.join(m_path, f))]
        for d in day_list:
            data[m][d] = []
            d_path = os.path.join(m_path, d)
            individual_list = [f for f in os.listdir(d_path) if os.path.isfile(os.path.join(d_path, f))]
            for i in individual_list:
                i_data = np.load(os.path.join(d_path, i))
                # Remove unix time from representation, probably better just to not save unix time after sorting, will fix later
                data[m][d].append(i_data[:, 1:])
    return data
