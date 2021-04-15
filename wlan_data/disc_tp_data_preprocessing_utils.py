import numpy as np
import pandas as pd
import datetime
import os


def get_as_num_list(list_as_string):
  nums = []
  for i in list_as_string.split(","):
    nums.append(int(i))
  return nums


def get_building_data(raw_data_directory):

  directory = raw_data_directory
  num_users = 0
  locs = []
  tmpstmps = []
  userids = []

  for filename in os.listdir(directory):
      if filename.endswith(".csv"):
        #print("************* " + filename + " *************")

        df = pd.read_csv(directory +"/"+filename, header=None, sep='delimeter')
        data = pd.DataFrame({'USERID' : [], 'SPACEID': [], 'TIMESTAMP': []})
        idx = 0
        for row in df.iterrows():
          if idx == df.shape[0]:
            continue
          else:
            curr_locations = get_as_num_list(df.loc[idx].tolist()[0])
            locs.extend(get_as_num_list(df.loc[idx].tolist()[0]))
            tmpstmps.extend(get_as_num_list(df.loc[idx + 1].tolist()[0]))
            userids.extend([num_users]*len(curr_locations))
          idx = idx + 2
          num_users = num_users + 1

  data.USERID = userids
  data.SPACEID = locs 
  data.TIMESTAMP = tmpstmps

  return data

'''
function to remove all non-changing adjacent points in a list
'''
def remove_stationary(list1, indices_list1):
  i = 0
  while i < (len(list1) - 1):
      # Modifying this to keep only biggest building data points
      if list1[i] == list1[i+1] or list1[i] <= 30 or list1[i] == 147 or list1[i + 1] == 147:
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

def sequence_train_test_split(df, length):
    user_sample_pool = []
    train_users = []
    test_users = []
    for user, df_user in df.groupby('USERID'):
        df_len = len(df_user)
        if df_len < 3 * length:
            train_users.append(df_user)
        else:
            user_sample_pool.append(df_user)

    user_index = max(df['USERID'].tolist()) + 1
    total_len = len(df)
    test_len = 0
    while test_len < 0.1 * total_len:
        df_idx = np.random.choice(len(user_sample_pool))
        sample_df = user_sample_pool.pop(df_idx)
        seq_idx = np.random.choice(np.arange(length, len(sample_df) - length))
        a = sample_df.iloc[0:seq_idx]
        if len(a) < 3 * length:
            train_users.append(a)
        else:
            user_sample_pool.append(a)

        b = sample_df.iloc[seq_idx:seq_idx + length]
        b.loc[:, 'USERID'] = user_index
        test_users.append(b)
        test_len += len(b)
        user_index += 1

        c = sample_df.iloc[seq_idx + length:]
        c.loc[:, 'USERID'] = float(user_index)
        if len(c) < 3 * length:
            train_users.append(c)
        else:
            user_sample_pool.append(c)
        user_index += 1

    train_users.extend(user_sample_pool)
    return pd.concat(train_users), pd.concat(test_users)

def random_sample_data(df, length, num_per_user):
    data = []
    for user, df_user in df.groupby('USERID'):
        df_len = len(df_user)
        mat = df_user.values[:, 2:]

        if df_len < length * num_per_user:
            continue
        indices = np.random.choice(df_len - length + 1, size=num_per_user, replace=False)
        for i in indices:
            data.append(mat[i:i+length])
    return data

'''
function to normalize lat/long coords
'''
def norm(x, mean, std):
  SCALE_FACTOR = 1
  val_normed = ( x - mean) / std * SCALE_FACTOR
  return val_normed

def spaceid_to_one_hot(df):
    # Gather all unique spaceids
    ids = set()
    for space_id in df['SPACEID'].tolist():
        ids.add(space_id)
    n_ids = len(ids)
    # Get arbitrary mapping between spaceids and indices for one hot encoding
    id_to_int = {}
    int_to_id = {}
    for i, x in enumerate(ids):
        int_to_id[i] = x
        id_to_int[x] = i
    # One hot encode and put in new dataframe
    one_hot_data = []
    for i, x in df.iterrows():
        data = np.zeros((n_ids + 2,))
        data[0] = x['USERID']
        data[1] = x['TIMESTAMP']
        data[id_to_int[x['SPACEID']] + 2] = 1
        one_hot_data.append(data)
    one_hot_df = pd.DataFrame(data=one_hot_data)
    one_hot_df.rename(columns={0:'USERID', 1:'TIMESTAMP'}, inplace=True)
    return one_hot_df, int_to_id, id_to_int

'''
gets the duration spent in space as seconds. Returns in new dataframe column
'''

def get_durations(data):

  durations_all_users = []
  for user, df_user in data.groupby('USERID'):

    durations_in_space = []
    curr_user_timestamps = df_user['TIMESTAMP'].tolist()

    for i, timestamp in enumerate(curr_user_timestamps):

      if i == len(curr_user_timestamps) - 1:
        curr_duration = "END"
      else:
        time_start = datetime.datetime.fromtimestamp(curr_user_timestamps[i])
        time_end = datetime.datetime.fromtimestamp(curr_user_timestamps[i+1] )
        time_difference = time_end - time_start
        curr_duration = time_difference.total_seconds()

      durations_in_space.append(curr_duration)
    
    durations_all_users.extend(durations_in_space)

  data['DURATION_IN_SPACE_SECONDS'] = durations_all_users

  return data

'''
splits existing users based on large time gaps as indicated by timegap_seconds_thresh input.
Returns dataframe with adjusted userIDs
'''
def split_trajectories(data, timegap_seconds_thresh, num_users):

  appended_df = []
  for user, df_user in data.groupby('USERID'):

    # with pd.option_context('display.max_rows', None):  # more options can be specified also
    #   display(df_user)

    idx_splits = []

    curr_user_timestamps = df_user['TIMESTAMP'].tolist()

    for i, timestamp in enumerate(curr_user_timestamps):

      if i == len(curr_user_timestamps) - 1:
        continue
      else:
        time_start = datetime.datetime.fromtimestamp(curr_user_timestamps[i])
        time_end = datetime.datetime.fromtimestamp(curr_user_timestamps[i+1] )
        time_difference = time_end - time_start
        curr_duration = time_difference.total_seconds()

      if curr_duration >= timegap_seconds_thresh:
        idx_splits.append(i+1) # append curr end time

    for i, idx in enumerate(idx_splits):

      if i == len(idx_splits) - 1:
        user_len = len(curr_user_timestamps)
        df_user.iloc[idx:user_len]['USERID'] = num_users
        num_users = num_users + 1
      else:
        df_user.iloc[idx:idx_splits[i+1]]['USERID'] = num_users
        num_users = num_users + 1

    appended_df.append(df_user)

  df_split_users = pd.concat(appended_df)
  df_split_users = df_split_users.sort_values(by=['USERID', 'TIMESTAMP'])
  return df_split_users

'''
leave one out cv split by user
'''

from sklearn.model_selection import LeaveOneOut

def get_cv_splits(data):

  cv_splits = []
  user_dfs = []

  for user, df_user in data.groupby('USERID'):
    user_dfs.append(df_user)


  loo = LeaveOneOut()
  for train_index, test_index in loo.split(np.array(user_dfs)):
    print("train idx: {}, test idx: {}".format(train_index, test_index))
    X_train, X_test = np.array(user_dfs)[train_index], np.array(user_dfs)[test_index]
    curr_split = [X_train, X_test]
    cv_splits.append(curr_split)

  return cv_splits


if __name__ == "__main__":
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

    TIMEGAP_THRESH = 3600 # in seconds
    NUM_USERS = 19
    df_sorted_transitions = split_trajectories(df_sorted_transitions, TIMEGAP_THRESH, NUM_USERS)
    # one_hot_df, int_to_id, id_to_int = spaceid_to_one_hot(df_sorted_transitions)

    # # # get duration in space
    # # df_sorted_transitions = get_durations(df_sorted_transitions)
    # # df_sorted_transitions['DURATION_IN_SPACE_MINUTES'] = df_sorted_transitions['DURATION_IN_SPACE_SECONDS'].apply(lambda x: x/60 if type(x) is float else 'END')
    
    # # # append duration as feature to one hot encoded df
    # # one_hot_df['100'] = df_sorted_transitions['DURATION_IN_SPACE_MINUTES'].tolist()
    # # display(one_hot_df)
