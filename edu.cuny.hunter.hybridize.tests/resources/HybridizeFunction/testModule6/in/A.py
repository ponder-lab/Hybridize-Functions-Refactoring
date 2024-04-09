# Test https://github.com/wala/ML/issues/65.

from tensorflow import ones
import B

B.f(ones([1, 2]))
