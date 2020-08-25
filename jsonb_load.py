import pbjson
with open('data.jsonb', "rb") as json_file:
    data = pbjson.load(json_file)
    print(len(data), len(data[0]))
