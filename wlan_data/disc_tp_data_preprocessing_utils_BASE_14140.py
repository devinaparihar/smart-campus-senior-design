import numpy as np
import pandas as pd


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
