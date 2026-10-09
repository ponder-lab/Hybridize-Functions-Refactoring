x = None


def leaky_function(a):
    global x
    x = [a]  # Stores a list created here into a global.


leaky_function(1)
print(x)
