import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import load

data = load.load_from_dir(sys.argv[1])

"""
for k in data.keys():
    print("Month: " + str(k))
    for d in data[k].keys():
        print("\tDay " + str(d) + ": " + str(len(data[k][d])))
        for x in data[k][d]:
            print(x)
"""

individual_data = load.compile_data_by_individual(data["March"])

count = 0
ids = []
times = []
lats = []
longs = []
for i in individual_data.keys():
    if len(individual_data[i]) < 1000:
        continue
    data = individual_data[i]
    indices = np.random.choice(a, size=1000, replace=False)
    ids.append(np.array([i for _ in range(1000)]))
    times.append(data[indices, 0])
    lats.append(data[indices, 1])
    longs.append(data[indices, 2])
    count += 1

print("Count: {}".format(count))

limit = 1000
columns = ["id", "time", "lat", "long"]
small_data = np.concatenate([np.concatenate(ids[:limit]).reshape(-1, 1), np.concatenate(times[:limit]).reshape(-1, 1), np.concatenate(lats[:limit]).reshape(-1, 1), np.concatenate(longs[:limit]).reshape(-1, 1)], axis=1)
print(small_data.shape)
small_df = pd.DataFrame(data=small_data, columns=columns)
print(small_df)
small_df.to_csv("1000.csv")

large_data = np.concatenate([np.concatenate(ids).reshape(-1, 1), np.concatenate(times).reshape(-1, 1), np.concatenate(lats).reshape(-1, 1), np.concatenate(longs).reshape(-1, 1)], axis=1)
print(large_data.shape)
large_df = pd.DataFrame(data=large_data, columns=columns)
print(large_df)
large_df.to_csv("bigger1000.csv")
