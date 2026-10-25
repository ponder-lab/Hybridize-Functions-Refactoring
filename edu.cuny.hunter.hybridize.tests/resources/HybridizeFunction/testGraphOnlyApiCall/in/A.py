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
def compat_gradients_step():
    return tf.compat.v1.gradients(v * v, [v])


@tf.function
def double():
    return v * 2.0


x_data = tf.ones([2, 3])


class Regressor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        return self.dense(inputs)


class NamedOptimizerRegressor(Regressor):
    @tf.function
    def gradients_step(self):
        loss = tf.reduce_mean(self(x_data))
        return self.optimizer.get_gradients(loss, self.trainable_variables)


class InstanceOptimizerRegressor(Regressor):
    @tf.function
    def gradients_step(self):
        loss = tf.reduce_mean(self(x_data))
        return self.optimizer.get_gradients(loss, self.trainable_variables)


def named_optimizer_step():
    model = NamedOptimizerRegressor()
    model.compile(optimizer="sgd", loss="mse")
    model(x_data)
    return model.gradients_step()


def instance_optimizer_step():
    model = InstanceOptimizerRegressor()
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss="mse")
    model(x_data)
    return model.gradients_step()


average_checkpoints()
get_updates_step()
gradients_step()
backend_gradients_step()
calls_gradients_helper()
double()
compat_gradients_step()
named_optimizer_step()
instance_optimizer_step()
