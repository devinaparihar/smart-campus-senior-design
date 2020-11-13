import numpy as np
import matplotlib.pyplot as plt

def rand_circle_points():
    radius = np.random.uniform(1, 5)
    noise_std = radius / 10
    direction = np.random.binomial(1, 0.5)
    center = np.array([np.random.uniform(-10, 10), np.random.uniform(-10, 10)])
    division = np.random.randint(10, 20)
    start_radians = np.random.uniform(0, 2 * np.pi)
    l = []
    increment = 2 * np.pi / division if direction == 1 else -2 * np.pi / division
    for i in range(division):
        radians = start_radians + increment * i
        point = center + np.array([np.cos(radians) * radius, np.sin(radians) * radius])
        l.append(point + np.random.normal(scale=noise_std, size=2))
    return np.stack(l)

"""
plt.figure()
for i in range(5):
    points = rand_circle_points()
    plt.scatter(points[:,0].ravel(), points[:,1].ravel())
plt.show()
"""

X = []
for i in range(3000):
    X.append(rand_circle_points())

np.save("circles_train.npy", X[:2000])
np.save("circles_test.npy", X[2000:2500])
np.save("circles_validate.npy", X[2500:])
