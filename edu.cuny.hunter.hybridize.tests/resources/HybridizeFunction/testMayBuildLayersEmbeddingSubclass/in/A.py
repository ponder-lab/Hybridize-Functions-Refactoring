import tensorflow as tf


class TFEmbedding(tf.keras.layers.Embedding):

    def call(self, inputs):
        return tf.gather(self.embeddings, inputs)


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.emb = TFEmbedding(10, 4)

    def call(self, x):
        return self.emb(x)


model = Net()
model(tf.constant([[1, 2]]))
