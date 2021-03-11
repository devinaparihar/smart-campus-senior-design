import numpy as np
import pandas as pd
import networkx as nx
import datetime as dt
import matplotlib.pyplot as plt

from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score

df = pd.read_csv("TrainingData.csv")

graph = nx.Graph()
#graph.add_nodes_from(range(1, df['SPACEID'].max() + 1))

MAX_SECONDS = 60
transitions = {}
data = {}
mappings = {}
reverse_mappings = {}

def increment(dic, key):
    if key not in dic.keys():
        dic[key] = 0
    dic[key] += 1

def add_mapping(mapping, reverse_mapping, location_id):
    if location_id not in mapping.keys():
        label = len(mapping.keys())
        mapping[location_id] = label
        reverse_mapping[label] = location_id

def add_data(dic, mappings, reverse_mappings, current, goal, features):
    if current not in dic.keys():
        dic[current] = ([], [])
        mappings[current] = {}
        reverse_mappings[current] = {}
    mapping = mappings[current]
    reverse_mapping = reverse_mappings[current]
    add_mapping(mapping, reverse_mapping, features[0])
    add_mapping(mapping, reverse_mapping, goal)
    dic[current][0].append(features)
    dic[current][1].append(mapping[goal])

DIVISIONS = 6

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
            increment(transitions, spaces[i])
            increment(transitions, spaces[i + 1])
            time = dt.datetime.fromtimestamp(times[i])
            add_data(data, mappings, reverse_mappings, spaces[i], spaces[i + 1], [condition, time.hour // DIVISIONS, time.weekday()])

            condition = spaces[i]
            n_transitions += 1

def add_to_list(dic, key, val):
    if key not in dic.keys():
        dic[key] = []
    dic[key].append(val)

def sample_base(y):
    dic = {}
    for i, y_i in enumerate(y):
        add_to_list(dic, y_i, i)
    indices = []
    for key in dic.keys():
        indices.append(np.random.choice(dic[key]))
    return set(indices)

def train_covers_test(train, test):
    indices = []
    for i, y_i in enumerate(test):
        if y_i not in train:
            indices.append(i)
    return indices

def one_hot(mapping, space_id):
    vec = np.zeros((len(mapping.keys()),))
    vec[mapping[space_id]] = 1
    return vec

def time_one_hot(value, total):
    vec = np.zeros((total,))
    vec[value] = 1
    return vec

NUM_FOLDS = 25

for smoothing in [0.1, 0.25, 0.5, 1, 2.5, 5]:
    n_better = 0
    n_ce_better = 0
    count = 0

    n_gram_correct = 0
    nb_correct = 0
    total = 0

    total_degree = 0

    for key in data.keys():
        mapping = mappings[key]
        X = []
        for x in data[key][0]:
#            X.append(np.concatenate([one_hot(mapping, x[0]), time_one_hot(x[1], DIVISIONS)]))
#            X.append(one_hot(mapping, x[0]))
            X.append(np.concatenate([one_hot(mapping, x[0]), time_one_hot(x[1], DIVISIONS), time_one_hot(x[2], 7)]))
        X = np.array(X)
        y = np.array(data[key][1])

        full_model = BernoulliNB(alpha=smoothing)
        full_model.fit(X, y)

        n_classes = len(full_model.classes_)
        total_degree += n_classes

        accuracies = []
        n_gram_accuracies = []
        losses = []
        n_gram_losses = []
        test_sizes = []
        train_sizes = []

        for i in range(NUM_FOLDS):
            X_train = []
            X_test = []
            y_train = []
            y_test = []
            for j in range(y.shape[0]):
                if np.random.uniform() < .9:
                    X_train.append(X[j, :])
                    y_train.append(y[j])
                else:
                    X_test.append(X[j, :])
                    y_test.append(y[j])

            indices = train_covers_test(y_train, y_test)
            for index in sorted(indices, reverse=True):
                total += 1
                del X_test[index]
                del y_test[index]
            if len(X_test) == 0 or len(X_train) == 0:
                continue

            X_train = np.stack(X_train, axis=0)
            X_test = np.stack(X_test, axis=0)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            train_sizes.append(X_train.shape[0])
            test_sizes.append(X_test.shape[0])

            model = BernoulliNB(alpha=smoothing)
            model.fit(X_train, y_train)
            n_gram_probas = []
            prior = np.exp(model.class_log_prior_)
            labels = model.classes_
            if labels.shape[0] == 1:
                labels = np.concatenate([labels, np.array([100])], axis=0)

            for i in range(X_test.shape[0]):
                n_gram_probas.append(prior)
            n_gram_probas = np.stack(n_gram_probas, axis=0)
            y_probs = model.predict_proba(X_test)
            losses.append(log_loss(y_test, y_probs, labels=labels))
            acc = model.score(X_test, y_test)
#            print("NB prediction: {}".format(model.predict(X_test)))
            accuracies.append(acc)
            n_gram_losses.append(log_loss(y_test, n_gram_probas, labels=labels))
            n_gram_predict = np.ones((X_test.shape[0],)) * model.classes_[np.argmax(prior)]
#            print("N-gram prediction: {}".format(n_gram_predict))
#            print("True labels: {}".format(y_test))
            n_gram_acc = accuracy_score(y_test, n_gram_predict)
            n_gram_accuracies.append(n_gram_acc)

            num = X_test.shape[0]
            total += num
            n_gram_correct += n_gram_acc * num
            nb_correct += acc * num



        if np.array(train_sizes).mean() > 10:
            count += 1
            if np.array(accuracies).mean() >= np.array(n_gram_accuracies).mean():
                n_better += 1
            if np.array(losses).mean() < np.array(n_gram_losses).mean():
                n_ce_better += 1

            """
            print("\n" + str(key))
            print("Num Possible Paths: {}".format(n_classes))
            print("Class Probabilities: {}".format(np.exp(full_model.class_log_prior_)))
            print("Feature Probabilities: {}".format(np.exp(full_model.feature_log_prob_)))
            if len(accuracies) > 0:
                print("Num Trials: {}".format(len(accuracies)))
                print("Average Train Size: {}".format(np.array(train_sizes).mean()))
                print("Average Test Size: {}".format(np.array(test_sizes).mean()))
                print("NB Accuracies: {}".format(np.array(accuracies).mean()))
                print("N-gram Accuracies: {}".format(np.array(n_gram_accuracies).mean()))
#        print("NB Cross Entropy: {}".format(np.array(losses).mean()))
#        print("N-gram Cross Entropy: {}".format(np.array(n_gram_losses).mean()))
            else:
                print("Low samples, {} total".format(X.shape[0]))
            """

    print("Total NB Accuracy: {}".format(nb_correct / total))
    print("Total N-gram Accuracy: {}".format(n_gram_correct / total))

    print("Smoothing Variable: {}\nFraction NB better than N-gram (accuracy): {}\nFraction NB better than N-gram (cross entropy): {}".format(smoothing, n_better / count, n_ce_better / count))
    print("Count: {}".format(count))
    print(total_degree / len(data.keys()))
    print(len(data.keys()))

"""
plt.figure()
nx.draw_spring(graph, with_labels=True, node_size=100)
plt.show()
"""
