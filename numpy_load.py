import pbjson
import numpy as np
import time

start = time.time()
with open("data.numpy", "rb") as f:
    data = np.fromfile(f)
    print(data.shape, time.time() - start)
