import pbjson

with open("data.jsonb", "rb") as f:
    data = pbjson.load(f)
    print(len(data), len(data[0]))
