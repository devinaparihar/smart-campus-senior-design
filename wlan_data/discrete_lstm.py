import torch
import pickle
import pandas as pd

from torch import nn

from disc_tp_data_preprocessing_utils import get_transitions_only, spaceid_to_one_hot, random_sample_data

"""
df = pd.read_csv("TrainingData.csv")
df_sorted = df.sort_values(by=['TIMESTAMP', 'USERID'])
df_sorted_transitions = get_transitions_only(df_sorted)

one_hot_df, int_to_id, id_to_int = spaceid_to_one_hot(df_sorted_transitions)

one_hot_df.to_csv("OneHot.csv", index=False)
with open('int_to_id.pickle', 'wb') as handle:
    pickle.dump(int_to_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('id_to_int.pickle', 'wb') as handle:
    pickle.dump(id_to_int, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""

df = pd.read_csv("OneHot.csv")
samples = random_sample_data(df, 10, 5)


