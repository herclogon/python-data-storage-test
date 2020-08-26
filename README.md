# Benchmark load matrix(10000, 10) with float values

Run in the command line:
```
time json_load.py
time jsonb_load.py
time numpy_load.py
time p7type_load.py
```

## Results

* **JSON**
  * file size: 2Mb
  * load time: 0.02s
* **JSONB**
  * file size: 975Kb
  * load time: 0.03s
* **Numpy**
  * file size: 782Kb
  * load time: 0.0004s
* **p7 typesystem**
  * file size: 787Kb
  * load time: 0.004s
* **In memory**
  * 781Kb
