import torch
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from torch import nn, optim
from copy import deepcopy

from disc_tp_data_preprocessing_utils import get_transitions_only, spaceid_to_one_hot, random_sample_data, sequence_train_test_split, split_trajectories, get_durations, get_building_data
from discrete_lstm import DeepLSTM, location_forward_pass_batch, time_forward_pass_batch, numpy_data_to_pytorch

IMAGE_DIR = './images/'

# Create Graph
df = pd.read_csv("TrainingData.csv")
graph = nx.Graph()

for user in range(1, df['USERID'].max() + 1):
    user_df = df[df['USERID'] == user].sort_values('TIMESTAMP')
    spaces = user_df['SPACEID'].to_numpy().ravel()

    condition = spaces[0]
    for i in range(1, spaces.shape[0] - 1):
        if spaces[i] > 30 and spaces[i] != 147 and spaces[i + 1] != 147 and spaces[i] != condition and spaces[i] != spaces[i + 1]:
            graph.add_edge(spaces[i], spaces[i + 1])
            condition = spaces[i]

pos = nx.spring_layout(graph, k=1/2)
nx.draw(graph, pos=pos, node_size=100)
plt.savefig(IMAGE_DIR + "original.png")

TIMEGAP_THRESH = 3600
NUM_USERS = 19

df = pd.read_csv("TrainingData.csv")
df_sorted = df.sort_values(by=['TIMESTAMP', 'USERID'])
df_sorted_transitions = split_trajectories(get_transitions_only(df_sorted), TIMEGAP_THRESH, NUM_USERS)
one_hot_df, int_to_id, id_to_int = spaceid_to_one_hot(df_sorted_transitions)

# get duration in space
df_sorted_transitions = get_durations(df_sorted_transitions)
# NEED TO DO SOMETHING ABOUT THESE NON FLOAT THINGS AT SOME POINT
df_sorted_transitions['DURATION_IN_SPACE_MINUTES'] = df_sorted_transitions['DURATION_IN_SPACE_SECONDS'].apply(lambda x: x/60 if type(x) is float else 1)

# append duration as feature to one hot encoded df
one_hot_df['STAYING_TIME'] = df_sorted_transitions['DURATION_IN_SPACE_MINUTES'].tolist()

one_hot_df.to_csv("OneHot.csv", index=False)
with open('int_to_id.pickle', 'wb') as handle:
    pickle.dump(int_to_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('id_to_int.pickle', 'wb') as handle:
    pickle.dump(id_to_int, handle, protocol=pickle.HIGHEST_PROTOCOL)

N_EPOCHS = 500
TRAIN_LENGTH = 10
N_PER_USER = 5
CUDA = torch.cuda.is_available()

df = pd.read_csv("OneHot.csv")

HIDDEN_STATE_SIZE = 100
N_LAYERS = 3
N_LOCATIONS = df.values.shape[1] - 3
INPUT_DIM = N_LOCATIONS + 1

LOCATION_LR = 0.01
TIME_LR = 0.005


test_steps = [1, 2, 3, 4, 5]

train_df, test_df = sequence_train_test_split(df, TRAIN_LENGTH)

location_network = DeepLSTM(N_LAYERS, HIDDEN_STATE_SIZE, INPUT_DIM, N_LOCATIONS)
time_network = DeepLSTM(N_LAYERS, HIDDEN_STATE_SIZE, INPUT_DIM, 1)
location_optimizer = optim.Adam(location_network.parameters(), lr=LOCATION_LR)
time_optimizer = optim.Adam(time_network.parameters(), lr=TIME_LR)

for i in range(N_EPOCHS):
    print("Epoch: {}".format(i))

    # Train the location prediction network
    location_optimizer.zero_grad()
    x = np.array(random_sample_data(train_df, TRAIN_LENGTH, N_PER_USER))
    x = torch.from_numpy(x).float()
    location_loss = location_forward_pass_batch(location_network, x, N_LOCATIONS)
    location_loss.backward()
    location_optimizer.step()
    print("Location Loss: {}".format(location_loss.item()))

    # Train the time prediction network
    time_optimizer.zero_grad()
    time_loss = torch.tensor(0).float()
    total = 0
    for j, n in enumerate(test_steps):
        x = np.array(random_sample_data(train_df, n + 1, TRAIN_LENGTH // (n + 1)))
        x = torch.from_numpy(x).float()
        time_loss += time_forward_pass_batch(time_network, x, N_LOCATIONS, n)
        total += x.size(0)
    time_loss.backward()
    time_optimizer.step()
    print("Average Time Loss: {}".format(time_loss.item() / total), flush=True)

CUDA = torch.cuda.is_available()

NUM_TO_VIS = 3
VIS_INIT_LEN = 5
VIS_FINAL_LEN = 10
N_EXAMPLES = 5
TIME_RESOLUTION = .1

EXPAND_TIME = 4
EXPAND_INTERVAL = 0.1
EXPAND_PROB = 0.8

int_to_id = pickle.load(open('int_to_id.pickle', 'rb'))

SOFTMAX = nn.Softmax(dim=-1)

nodes_list = graph.nodes()
node_to_ind = {node: i for i, node in enumerate(nodes_list)}

with torch.no_grad():
    for example in range(N_EXAMPLES):
        current_time = 0
        print("Example {}".format(example))
        rs = random_sample_data(test_df, VIS_INIT_LEN, 1)
        data = np.array(rs)[np.random.choice(len(rs), size=NUM_TO_VIS, replace=False).tolist()]

        times = data[:, VIS_INIT_LEN - 1, N_LOCATIONS]
        new_times = [np.random.uniform(low=0, high=t) for t in times]
        data[:, VIS_INIT_LEN - 1, N_LOCATIONS] = np.array(new_times)

        coeffs = [1 / NUM_TO_VIS for i in range(NUM_TO_VIS)]

        output_times = time_network.forward_next_step(torch.from_numpy(data).float()).numpy().ravel()

        remaining_times = [(max(0, output_times[i]), data[i]) for i in range(NUM_TO_VIS)]
        remaining_times.sort(key=lambda x : x[0])

        while current_time < EXPAND_TIME:
            print("Current time: {}".format(current_time))
            print("Times: {}".format([tup[0] for tup in remaining_times]))
            print("Lengths: {}".format([len(tup[1]) for tup in remaining_times]))
            print("Coefficients: {}".format(coeffs), flush=True)
            print("Sum of coefficients: {}".format(sum(coeffs)))
            remove_time = remaining_times[0][0]
            if remove_time <= EXPAND_INTERVAL:
                current_time += remove_time
                overlap = EXPAND_INTERVAL - remove_time
                seq = remaining_times[0][1]
                coeff = coeffs[0]

                del remaining_times[0]
                del coeffs[0]
                remaining_times = [(remaining_times[i][0] - remove_time, remaining_times[i][1]) for i in range(len(remaining_times))]

                next_location_probs = SOFTMAX(location_network.forward_next_step(torch.from_numpy(seq).float().unsqueeze(0))).numpy().ravel()
                weight = 0
                index_list = np.argsort(next_location_probs * -1)
                list_loc = 0
                temp_coeffs = []
                while weight < EXPAND_PROB:
                    new_seq = deepcopy(seq)
                    new_transition = np.zeros(INPUT_DIM)
                    new_transition[index_list[list_loc]] = 1
                    new_transition[N_LOCATIONS] = overlap
                    new_seq = np.concatenate([new_seq, [new_transition]], axis=0)
                    new_time = max(0, time_network.forward_next_step(torch.from_numpy(new_seq).float().unsqueeze(0)).item())
                    new_seq[-1, N_LOCATIONS] += new_time
                    remaining_times.append((overlap + new_time, new_seq))
                    temp_coeffs.append(next_location_probs[index_list[list_loc]])
                    weight += next_location_probs[index_list[list_loc]]
                    list_loc += 1
                coeffs.extend([coeff * temp / sum(temp_coeffs) for temp in temp_coeffs])
                new_order = np.argsort([tup[0] for tup in remaining_times])
                remaining_times = [remaining_times[i] for i in new_order]
                coeffs = [coeffs[i] for i in new_order]
            else:
                remaining_times = [(remaining_times[i][0] - EXPAND_INTERVAL, remaining_times[i][1]) for i in range(len(remaining_times))]
                current_time += EXPAND_INTERVAL

            color_params = [0 for i in range(len(nodes_list))]
            total = sum(coeffs)
            for j, tup in enumerate(remaining_times):
                v = np.argmax(tup[1][-1][:N_LOCATIONS])
                color_params[node_to_ind[int_to_id[v]]] += coeffs[j] / total
            colors = [(np.clip(2 * (1 - x), 0, 1), np.clip(2 * x, 0, 1), 0) if x != 0 else (0, 0, 0) for x in color_params]

            nx.draw(graph, pos=pos, nodelist=nodes_list, node_color=colors, node_size=100)
            plt.savefig(IMAGE_DIR + "{:02d}_{:03d}.png".format(example, int(np.floor(current_time / EXPAND_INTERVAL))))
