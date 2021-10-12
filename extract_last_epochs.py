import numpy as np


acc = np.array([[82.6496, 83.2042, 80.3080],
                [96.4260, 96.6285, 95.2904],
                [82.1126, 81.7693, 82.4823],
                [96.0739, 96.0387, 96.2147],
                [81.2588, 79.1813, 80.4489],
                [96.4876, 95.4489, 95.4137]])
mean = np.mean(acc, 1)
var = np.var(acc, 1)
print(mean, var)