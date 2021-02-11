import numpy as np
import pandas as pd
import networkx as nx
import datetime as dt
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("TrainingData.csv")

graph = nx.Graph()
#graph.add_nodes_from(range(1, df['SPACEID'].max() + 1))

MAX_SECONDS = 60
transitions = {}
data = {}
mappings = {}
reverse_mappings = {}

def increment_transition(dic, key):
    if key not in dic.keys():
        dic[key] = 0
    dic[key] += 1

def add_mapping(mapping, reverse_mapping, location_id):
    if location_id not in mapping.keys():
        label = len(mapping.keys())
        mapping[location_id] = label
        reverse_mapping[label] = location_id

def add_data(dic, mappings, reverse_mappings, current, prev, goal, weekday, hour, quarter):
    if current not in dic.keys():
        dic[current] = ([], [])
        mappings[current] = {}
        reverse_mappings[current] = {}
    mapping = mappings[current]
    reverse_mapping = reverse_mappings[current]
    add_mapping(mapping, reverse_mapping, prev)
    add_mapping(mapping, reverse_mapping, goal)
    dic[current][0].append([mapping[prev], weekday, hour, quarter])
    dic[current][1].append([mapping[goal]])

n_transitions = 0
for user in range(1, df['USERID'].max() + 1):
    user_df = df[df['USERID'] == user].sort_values('TIMESTAMP')
    times = user_df['TIMESTAMP'].to_numpy().ravel()
    spaces = user_df['SPACEID'].to_numpy().ravel()

    condition = spaces[0]
    for i in range(1, times.shape[0] - 1):
        diff = dt.datetime.fromtimestamp(times[i + 1]) - dt.datetime.fromtimestamp(times[i])
        if spaces[i] > 30 and spaces[i] != 147 and spaces[i + 1] != 147 and spaces[i] != condition and spaces[i] != spaces[i + 1] and diff.total_seconds() <= MAX_SECONDS:
            graph.add_edge(spaces[i], spaces[i + 1])
            increment_transition(transitions, spaces[i])
            increment_transition(transitions, spaces[i + 1])
            time = dt.datetime.fromtimestamp(times[i])
            add_data(data, mappings, reverse_mappings, spaces[i], condition, spaces[i + 1], time.weekday(), time.hour, time.minute // 15)

            condition = spaces[i]
            n_transitions += 1

print(transitions)
print(n_transitions)

for key in data.keys():
    print(np.array(data[key][0]))
    print(np.array(data[key][1]))

plt.figure()
nx.draw_spring(graph, with_labels=True, node_size=100)
plt.show()
