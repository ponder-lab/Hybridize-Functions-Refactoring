import tensorflow as tf

v = tf.Variable(1.0)
optimizer = tf.keras.optimizers.SGD()


def average_checkpoints():
    model = tf.keras.layers.Dense(3)
    model.build([None, 4])
    adam = tf.keras.optimizers.Adam()

    @tf.function
    def _build_model():
        x = tf.random.uniform([4, 4])
        y = model(x)
        loss = tf.reduce_mean(y)
        gradients = adam.get_gradients(loss, model.trainable_variables)
        adam.apply_gradients(zip(gradients, model.trainable_variables))

    _build_model()


@tf.function
def get_updates_step():
    optimizer.get_updates(v * v, [v])


@tf.function
def gradients_step():
    return tf.gradients(v * v, [v])


@tf.function
def backend_gradients_step():
    return tf.keras.backend.gradients(v * v, [v])


def gradients_of_square():
    return tf.gradients(v * v, [v])


@tf.function
def calls_gradients_helper():
    return gradients_of_square()


@tf.function
def double():
    return v * 2.0


average_checkpoints()
get_updates_step()
gradients_step()
backend_gradients_step()
calls_gradients_helper()
double()
