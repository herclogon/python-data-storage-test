import typesystem
import time

start = time.time()
with typesystem.File("data.p7type") as f:
    a = f.read()
    print(a.shape, time.time() - start)
