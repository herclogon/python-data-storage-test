import typesystem

with typesystem.File("data.p7type") as f:
    a = f.read()
    print(a.shape)
