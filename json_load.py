import json
import time

start = time.time()
with open("data.json") as f:
    data = json.load(f)
    print(len(data), len(data[0]), time.time() - start)
