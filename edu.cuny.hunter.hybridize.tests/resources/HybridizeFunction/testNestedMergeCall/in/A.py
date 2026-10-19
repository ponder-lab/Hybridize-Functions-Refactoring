import tensorflow as tf

cpu = tf.config.list_physical_devices("CPU")[0]
tf.config.set_logical_device_configuration(
    cpu, [tf.config.LogicalDeviceConfiguration()] * 2
)

strategy = tf.distribute.MirroredStrategy(["/cpu:0", "/cpu:1"])

with strategy.scope():
    w = tf.Variable([1.0, 2.0])
    optimizer = tf.keras.optimizers.SGD(0.1)


def apply_step(x):
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(w * x)
    gradients = tape.gradient(loss, [w])
    optimizer.apply_gradients(zip(gradients, [w]))
    return loss


def nested_step(x):
    return apply_step(x)


def direct_step(x):
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(w * x)
    gradients = tape.gradient(loss, [w])
    optimizer.apply_gradients(zip(gradients, [w]))
    return loss


def scale(x):
    return x * 2.0


def math_step(x):
    return tf.reduce_sum(scale(x))


x = tf.constant([1.0, 1.0])
strategy.run(nested_step, args=(x,))
strategy.run(direct_step, args=(x,))
strategy.run(math_step, args=(x,))
