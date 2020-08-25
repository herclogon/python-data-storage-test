import json

with open("data.json") as f:
    data = json.load(f)
    print(len(data), len(data[0]))
