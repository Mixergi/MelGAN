import tensorflow as tf

from ResStack import ResStack


class Generator(tf.keras.Model):
    def __init__(self, **kwargs):
        super(Generator, self).__init__(**kwargs)

        filters = 512

        upsample = [8, 8, 2, 2]

        self.blocks = [
            tf.keras.layers.Conv1D(filters, 7, 1, padding='same')
        ]

        for i, s in enumerate(upsample):
            self.blocks.extend([
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv1DTranspose(filters // (2**(i+1)), s*2, s, padding='same'),
                ResStack(filters // (2**(i+1)))
            ])

        self.blocks.extend([
            tf.keras.layers.Conv1D(1, 7, 1, padding='same'),
            tf.keras.layers.Activation(tf.nn.tanh)
        ])

    def call(self, x):

        for block in self.blocks:
            x = block(x)

        return x

    def export_model(self, save_dir, input_shape):
        input_layer = tf.keras.Input(input_shape)
        model = tf.keras.Model(input_layer, self.call(input_layer))
        model.save(save_dir)
