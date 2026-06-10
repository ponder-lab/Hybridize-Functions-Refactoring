# Fully-qualified-import variant of `testReconfigureBareDecorator`. Bare `import tensorflow` (no `as` alias) means the source-write
# must qualify every emitted name with the `tensorflow.` prefix when reconfiguring the existing `@tensorflow.function` decorator.
import tensorflow


@tensorflow.function(input_signature=[tensorflow.TensorSpec(shape=(), dtype=tensorflow.float32)])
def f(x):
    return x + 1


if __name__ == "__main__":
    f(tensorflow.constant(2.0))
