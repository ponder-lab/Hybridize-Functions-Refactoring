# From https://docs.python.org/3/library/io.html#io.IOBase.


def f():
  with open('spam.txt', 'w') as file:
    file.write('Spam and eggs!')


f()
