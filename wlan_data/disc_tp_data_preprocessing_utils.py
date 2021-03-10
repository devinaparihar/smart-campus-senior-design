import numpy as np
import pandas as pd


'''
function to remove all non-changing adjacent points in a list
'''
def remove_stationary(list1, indices_list1):
  i = 0
  while i < (len(list1) - 1):
      if list1[i] == list1[i+1]:
          del list1[i]
          del indices_list1[i]
      else:
          i = i + 1

'''
takes a sorted dataframe(by time) and returns a sorted dataframe (by user and time) 
that contains transitions only.
'''
def get_transitions_only(df_sorted):
  tot = 0
  df_indexes_to_keep = []
  appended_df = []
  for user, df_user in df_sorted.groupby('USERID'):

    spaceIDsIndexesList = df_user['SPACEID'].index.tolist()
    spaceIDsList = df_user['SPACEID'].tolist()

    # remove the non-changing location points and respective indices(stationary)
    remove_stationary(spaceIDsList, spaceIDsIndexesList)

    df_indexes_to_keep.extend(spaceIDsIndexesList)
    df_curr = df_user.loc[spaceIDsIndexesList]
    appended_df.append(df_curr)
    tot = tot + len(spaceIDsList)
    # print(len(spaceIDsList))
    # print(spaceIDsList)

  df_sorted_transitions = pd.concat(appended_df)

  return df_sorted_transitions

'''
Function to sample the data into sequences of specified length. 
sample_length is specified length (will be this length at most)
num_points_to_predict is the length of the sequence to be predicted.
Returns two lists - first returned list is X data, second list returned is target Y data
'''
def sample_data(data, sample_length, num_points_to_predict):

  X_dat = []
  y_dat = [] 

  for user, df_user in data.groupby('USERID'):
    sample_length = sample_length

    currlen = len(df_user['SPACEID'].tolist())

    if sample_length >= currlen:
      if currlen != 1:
        sample_length = currlen - 1
      else:
        continue #this user never moved anywhere --> skip

    sampled_instances = [df_user['SPACEID'].tolist()[x:x+sample_length] for x in range(0, currlen, sample_length)]

    sampled_instances = sampled_instances[0:-1] # remove last chunk since it is a leftover number (could change this to be a redistribution instead)

    for inst in sampled_instances:
      X_dat.append(inst[0:num_points_to_predict*-1])
      y_dat.append(inst[num_points_to_predict*-1:])

  return X_dat, y_dat

'''
function to normalize lat/long coords
'''
def norm(x, mean, std):
  SCALE_FACTOR = 1
  val_normed = ( x - mean) / std * SCALE_FACTOR
  return val_normed



# Example

# load data
train_df = pd.read_csv('TrainingData.csv')
validate_df = pd.read_csv('ValidationData.csv')

# concat and sort data
df = pd.concat([train_df, validate_df])
df_sorted = df.sort_values(by=['TIMESTAMP', 'USERID'])

# get normalized coordinates
long_mean = df_sorted['LONGITUDE'].mean(axis = 0)
lat_mean = df_sorted['LATITUDE'].mean(axis = 0)
long_std = df_sorted['LONGITUDE'].std(axis = 0)
lat_std = df_sorted['LATITUDE'].std(axis = 0)
df_sorted['LATITUDE_norm'] = df_sorted['LATITUDE'].apply(lambda x: norm(x, lat_mean, lat_std))
df_sorted['LONGITUDE_norm'] = df_sorted['LONGITUDE'].apply(lambda x: norm(x, long_mean, long_std))

SAMPLE_LENGTH = 20
NUM_POINTS_TO_PREDICT = 5

# get transitions only and sample
df_sorted_transitions = get_transitions_only(df_sorted)
X_dat, y_dat = sample_data(df_sorted_transitions, SAMPLE_LENGTH, NUM_POINTS_TO_PREDICT)