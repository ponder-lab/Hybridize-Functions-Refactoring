import tensorflow as tf


# Reduces TensorFlow2.0-Examples' CNN.py `train_step`: already hybrid with a bare
# `@tf.function`, two tensor parameters, no Python literal, and the variable collection read
# after the forward pass (the benign ordering of #822). That makes it the reconfigure
# population, where adding an inferred signature is its own transformation, and where a
# withheld signature costs the transformation rather than only the signature. It had no
# fixture when #890 could have swallowed exactly that, so it is pinned here.
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = tf.keras.layers.Dense(10)

    def call(self, x):
        return self.d1(x)


model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


train_step(tf.ones((2, 4)), tf.constant([1, 3]))
