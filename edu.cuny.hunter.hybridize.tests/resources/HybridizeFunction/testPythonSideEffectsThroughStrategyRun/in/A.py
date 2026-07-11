import tensorflow as tf


class Trainer:
    def __init__(self):
        self.dense = tf.keras.layers.Dense(2)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        self.loss = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.SUM
        )
        self.loss_value = None
        self.grad = None

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.dense(x)
            self.loss_value = self.loss(y, predictions)
        gradients = tape.gradient(self.loss_value, self.dense.trainable_variables)
        self.grad = gradients
        self.optimizer.apply_gradients(zip(gradients, self.dense.trainable_variables))
        return predictions

    def dist_train_step(self, strategy, x, y):
        return strategy.run(self.train_step, args=(x, y))


class LocalTrainer:
    def __init__(self):
        self.dense = tf.keras.layers.Dense(2)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
        self.loss = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.SUM
        )

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.dense(x)
            loss_value = self.loss(y, predictions)
        gradients = tape.gradient(loss_value, self.dense.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.dense.trainable_variables))
        return predictions

    def dist_train_step(self, strategy, x, y):
        return strategy.run(self.train_step, args=(x, y))


strategy = tf.distribute.MirroredStrategy()
x = tf.ones((4, 3))
y = tf.ones((4, 2))

trainer = Trainer()
trainer.dist_train_step(strategy, x, y)

local_trainer = LocalTrainer()
local_trainer.dist_train_step(strategy, x, y)
