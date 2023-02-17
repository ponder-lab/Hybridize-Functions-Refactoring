import tensorflow as tf
from tf_image.core.convert_type_decorator import convert_type

@tf.function
@convert_type
def test():
    pass

@convert_type
@tf.function
def test():
    pass
