'''
Created on Jun 28, 2022
@author: rk1424
'''

# Create an override model to classify pictures

import tensorflow as tf
from tensorflow.python.eager.def_function import function


@function  # This should be hybrid, even though we are not using tf.function. It's coming from the import statement above.
def test2():
    print("test2")
