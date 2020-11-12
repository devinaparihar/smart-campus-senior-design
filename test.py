import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

import extract

# Make csv size limit as big as possible on machine
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = max_int // 10

sample_size = 5000
columns = ["id", "time", "lat", "long"]
data = extract.sample_per_day("./raw_data/", sample_size)
df = pd.DataFrame(data=data, columns=columns)
df.to_csv("sampled_from_raw_5000_per_day.csv")
