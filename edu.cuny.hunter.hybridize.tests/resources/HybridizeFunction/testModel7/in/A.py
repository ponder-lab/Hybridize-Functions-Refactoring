import tensorflow as tf

# Create an override model to classify pictures


class SequentialModel(tf.keras.Model):

    def __init__(self, **kwargs):
        super(SequentialModel, self).__init__(**kwargs)

        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))

        # Add a lot of small layers
        num_layers = 100
        self.my_layers = [
            tf.keras.layers.Dense(64, activation="relu") for n in range(num_layers)
        ]

        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense_2 = tf.keras.layers.Dense(10)

        self._stuff = 5

    def call(self, x):
        x = self.flatten(x)

        for layer in self.my_layers:
            x = layer(x)

        x = self.dropout(x)
        x = self.dense_2(x)

        self._stuff = 6

        return x

    def get_stuff(self):
        return self._stuff


input_data = tf.random.uniform([20, 28, 28])
print("Input:")
print(type(input_data))
print(input_data)

model = SequentialModel()
print(model.get_stuff())

result = model.call(input_data)
print(model.get_stuff())

print("Output:")
print(type(input_data))
print(result)
