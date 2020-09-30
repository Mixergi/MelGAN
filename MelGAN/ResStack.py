import tensorflow as tf


class ResBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 dilation_rate=1,
                 initializer_seed=42,
                 **kwargs):
        super(ResBlock, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.initializer_seed = initializer_seed

    def build(self, input_shape):
        output_shape = input_shape[-1]

        self.layers = [
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv1D(self.filters,
                                   self.kernel_size,
                                   padding='same',
                                   dilation_rate=(1),
                                   name=f'conv_1 {self.kernel_size}x1 dilation=1'),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv1D(output_shape,
                                   self.kernel_size,
                                   padding='same',
                                   dilation_rate=(self.dilation_rate),
                                   name=f'conv_2 {self.kernel_size}x1 dilation={self.dilation_rate}')
        ]

    def call(self, inputs):

        x = inputs

        for layer in self.layers:
            x = layer(x)

        return inputs + x


class ResStack(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size=3,
                 dilation_rate=1,
                 m_conv=3,
                 n_residual=3,
                 initializer_seed=42,
                 **kwargs):
        super(ResStack, self).__init__(**kwargs)

        self.resblocks = [ResBlock(filters,
                                   kernel_size,
                                   int(dilation_rate * (m_conv ** i)),
                                   initializer_seed,
                                   name=f'ResBlock {kernel_size}x1 dilation={dilation_rate}')
                          for i, depth
                          in enumerate(range(n_residual))]

    def call(self, x):

        for resblock in self.resblocks:
            x = resblock(x)

        return x
