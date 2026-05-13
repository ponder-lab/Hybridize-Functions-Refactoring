# Scenario 1 of the input-signature inference algorithm: single call site, single dtype, single
# concrete shape. The trivial happy path of Algorithm 2—the inferred signature is exactly that
# single (dtype, shape) tuple, wrapped as a singleton InputSignature.

import tensorflow as tf


def func(t):
	return t + 1


func(tf.constant([1.0, 2.0]))
