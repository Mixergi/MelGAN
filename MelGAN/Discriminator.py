import tensorflow as tf


class Discriminator_Block(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Discriminator_Block, self).__init__(**kwargs)

        down_sampling = [4, 4, 4, 4]
        filters = [64, 256, 1024, 1024]

        self.blocks = [
            tf.keras.layers.Conv1D(16, 15, 1, padding='same'),
            tf.keras.layers.LeakyReLU()
        ]

        for i, (d, f) in enumerate(zip(down_sampling, filters)):
            self.blocks += [
                tf.keras.layers.Conv1D(
                    f, 41, d, padding='same', groups=d**(i+1)),
                tf.keras.layers.LeakyReLU()
            ]

        self.blocks += [
            tf.keras.layers.Conv1D(1024, 5, 1, padding='same'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv1D(1, 3, 1, padding='same')
        ]

    def call(self, x):
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i % 2 == 0:
                outputs.append(x)

        return outputs


class Discriminator(tf.keras.models.Model):
    def __init__(self, n_scale=3, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

        self.discriminator_blocks = [
            Discriminator_Block() for i in range(n_scale)]

        self.avgpooling = tf.keras.layers.AveragePooling1D(4)

    def call(self, inputs):
        outputs = []
        for block in self.discriminator_blocks:
            x = block(inputs)
            outputs.append(x)
            inputs = self.avgpooling(inputs)

        return outputs
