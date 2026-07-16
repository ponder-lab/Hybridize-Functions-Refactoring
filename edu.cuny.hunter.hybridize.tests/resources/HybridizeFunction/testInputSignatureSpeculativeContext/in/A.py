# Regression fixture for #783. `train_step` matches the speculative-context name regex, so
# `inferTensorParameters` sets the function-level tensor-parameter verdict from context once no
# parameter classifies. `config` points to a `Config` instance: not a tensor (`isTensor()` is
# FALSE, `getTensorTypes()` is empty) and not a primitive, so the P1 chain is not blocked by
# `HAS_PRIMITIVE_PARAMETERS` and inference runs. Before #783, `inferInputSignature` dispatched
# per parameter and reported `NON_TENSOR_PARAMETER`, contradicting the `SPECULATIVE_ANALYSIS`
# INFO the same pass emitted and advising a source change to code the tool had just judged
# tensor-typed by context.
import tensorflow as tf


class Config:
    pass


def train_step(config):
    return tf.reduce_sum(tf.constant([1.0, 2.0]))


train_step(Config())
