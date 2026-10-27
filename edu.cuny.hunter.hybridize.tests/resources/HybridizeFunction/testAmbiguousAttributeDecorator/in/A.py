import decs
import other


@decs.option
def f(x):
    return x


print(f(1))
print(other.option)
