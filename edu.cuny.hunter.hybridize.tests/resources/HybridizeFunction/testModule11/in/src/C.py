# Test https://github.com/wala/ML/issues/163.

from tensorflow import ones
from B import D

D().g(ones([1, 2]))
