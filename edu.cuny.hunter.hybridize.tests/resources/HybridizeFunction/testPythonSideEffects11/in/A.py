def f():
    with open("file.txt", "w") as f:
        f.writelines(["line1\n", "line2\n"])


f()
