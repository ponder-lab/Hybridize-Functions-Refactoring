# Test https://github.com/wala/ML/issues/65.

from tensorflow import ones
from B import f

f(ones([1, 2]))
