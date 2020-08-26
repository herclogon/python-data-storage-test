# Benchmark load matrix(10000, 10) with float values

## Install

```
poetry install
```

## Run

```
python json_load.py
python jsonb_load.py
python numpy_load.py
python p7type_load.py
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
