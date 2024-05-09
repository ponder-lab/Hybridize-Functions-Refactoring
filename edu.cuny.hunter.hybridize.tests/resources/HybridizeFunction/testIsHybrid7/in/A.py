# Same is testIsHybridTrue, except that we use "from" in the import statement.
from tensorflow import function


@function
def func():
    pass


if __name__ == "__main__":
    func()
