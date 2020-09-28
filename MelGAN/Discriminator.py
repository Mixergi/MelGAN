import tensorflow as tf


class Discriminator_Block(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Discriminator_Block, self).__init__(**kwargs)

    def call(self, x):
        pass


class Discriminator(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

    def call(self, x):
        pass