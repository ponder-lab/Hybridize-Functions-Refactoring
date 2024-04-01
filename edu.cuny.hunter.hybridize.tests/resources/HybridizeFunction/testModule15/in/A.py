# Test https://github.com/wala/ML/issues/163.

from tensorflow import ones
from src.C import D

D().f(ones([1, 2]))
