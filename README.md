# Test load matrix(10, 10000) with float values.

Run in the command line:
```
time json_load.py
time jsonb_load.py
time numpy_load.py
```

## Results

* **JSON**
  * file size: 2Mb
  * load time: 0.06s
* **JSONB**
  * file size: 966Kb
  * load time: 0.05s
* **Numpy**
  * file size: 800Kb
  * load time: 0.14s
* **In memory**
  * 781Kb



