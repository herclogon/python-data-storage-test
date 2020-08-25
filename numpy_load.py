import pbjson
import numpy as np

with open("data.numpy", "rb") as f:
    data = np.fromfile(f)
    print(data.shape)
