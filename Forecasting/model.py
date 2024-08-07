import tensorflow as tf
from tensorflow.keras import Model


class ConvBlock(tf.keras.Model):
    def __init__(self, filters):
        super().__init__()
        # Spatial decomposition
        self.conv1 = tf.keras.layers.Conv3D(filters,
                                            kernel_size=(3, 3, 1),
                                            strides=1,
                                            padding='same',
                                            use_bias=True,
                                            )
        # Temporal decomposition
        self.conv2 = tf.keras.layers.Conv3D(filters,
                                            kernel_size=(1, 1, 3),
                                            strides=1,
                                            padding='same',
                                            use_bias=True, )
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.relu_1 = tf.keras.layers.ReLU()

        self.conv3 = tf.keras.layers.Conv3D(filters,
                                            kernel_size=(3, 3, 1),
                                            strides=1,
                                            padding='same',
                                            use_bias=True,
                                            )
        # Temporal decomposition
        self.conv4 = tf.keras.layers.Conv3D(filters,
                                            kernel_size=(1, 1, 3),
                                            strides=1,
                                            padding='same',
                                            use_bias=True,)
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()

    def call(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.batch_norm_1(x)
        x = self.relu_1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batch_norm_2(x)
        x = tf.keras.activations.relu(x)
        return x

    def make(self, input_shape):
        '''
        This method makes the command "model.summary()" work.
        input_shape: (H,W,C), do not specify batch B
        '''
        x = tf.keras.layers.Input(shape=input_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name='actor')
        print(model.summary(line_length=120, show_trainable=True))
        return model


class UpConv(tf.keras.Model):
    def __init__(self, filters):
        super().__init__()
        self.up_1 = tf.keras.layers.UpSampling3D(size=(2, 2, 1))
        # Spatial decomposition
        self.conv1 = tf.keras.layers.Conv3D(filters,
                                            kernel_size=(3, 3, 1),
                                            strides=1,
                                            padding='same',
                                            use_bias=True,
                                            )
        # Temporal decomposition
        self.conv2 = tf.keras.layers.Conv3D(filters,
                                            kernel_size=(1, 1, 3),
                                            strides=1,
                                            padding='same',
                                            use_bias=True,)
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()

    def call(self, input):
        x = self.up_1(input)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch_norm_1(x)
        x = tf.keras.activations.relu(x)
        return x

    def make(self, input_shape):
        '''
        This method makes the command "model.summary()" work.
        input_shape: (H,W,C), do not specify batch B
        '''
        x = tf.keras.layers.Input(shape=input_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name='actor')
        print(model.summary(line_length=120, show_trainable=True))
        return model


class AttentionBlock(tf.keras.Model):
    def __init__(self, filters):
        super().__init__()
        self.conv_1 = tf.keras.layers.Conv3D(filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', use_bias=True)
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()

        self.conv_2 = tf.keras.layers.Conv3D(filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', use_bias=True)
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()

        self.conv_3 = tf.keras.layers.Conv3D(1, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', use_bias=True)
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()

    def call(self, g, x):
        g1 = self.conv_1(g)
        g1 = self.batch_norm_1(g1)

        x1 = self.conv_2(x)
        x1 = self.batch_norm_2(x1)

        psi = self.conv_3(g1 + x1)
        psi = self.batch_norm_3(psi)
        psi = tf.keras.activations.sigmoid(psi)
        psi = tf.keras.activations.relu(psi)
        return x*psi


class MyUnetModel(Model):
    def __init__(self, prediction_length, mask):
        super().__init__()
        # input_shape = (batch_size, 161, 181, 10, features_amount)
        self.mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        self.maxpool = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 1))

        self.conv_block_1 = ConvBlock(filters=64)
        self.conv_block_2 = ConvBlock(filters=128)
        self.conv_block_3 = ConvBlock(filters=256)
        self.conv_block_4 = ConvBlock(filters=512)
        self.conv_block_5 = ConvBlock(filters=1024)

        self.up_5 = UpConv(filters=512)
        self.attention_5 = AttentionBlock(filters=256)
        self.up_conv_5 = ConvBlock(filters=512)

        self.up_4 = UpConv(filters=256)
        self.attention_4 = AttentionBlock(filters=128)
        self.up_conv_4 = ConvBlock(filters=256)

        self.up_3 = UpConv(filters=128)
        self.attention_3 = AttentionBlock(filters=64)
        self.up_conv_3 = ConvBlock(filters=128)

        self.up_2 = UpConv(filters=64)
        self.attention_2 = AttentionBlock(filters=32)
        self.up_conv_2 = ConvBlock(filters=64)

        self.Conv_1x1 = tf.keras.layers.Conv3D(3, kernel_size=1, strides=1, padding='same')

    def call(self, x):
        # padding the image with zeroes to shape 192 x 192
        x1 = tf.pad(x, [[0, 0], [15, 16], [5, 6], [0, 0], [0, 0]])

        # encoding path
        x1 = self.conv_block_1(x1)
        # print(f'x1.shape = {x1.shape}')

        x2 = self.maxpool(x1)
        # print(x2.shape)
        x2 = self.conv_block_2(x2)
        # print(f'x2.shape = {x2.shape}')

        x3 = self.maxpool(x2)
        # print(x3.shape)
        x3 = self.conv_block_3(x3)
        # print(f'x3.shape = {x3.shape}')

        x4 = self.maxpool(x3)
        # print(x4.shape)
        x4 = self.conv_block_4(x4)
        # print(f'x4.shape = {x4.shape}')

        x5 = self.maxpool(x4)
        # print(x5.shape)
        x5 = self.conv_block_5(x5)
        # print(f'x5.shape = {x5.shape}')

        # decoding + concat path
        d5 = self.up_5(x5)
        x4 = self.attention_5(g=d5, x=x4)
        d5 = tf.concat((x4, d5), axis=4)
        d5 = self.up_conv_5(d5)

        d4 = self.up_4(d5)
        x3 = self.attention_4(g=d4, x=x3)
        d4 = tf.concat((x3, d4), axis=4)
        d4 = self.up_conv_4(d4)

        d3 = self.up_3(d4)
        x2 = self.attention_3(g=d3, x=x2)
        d3 = tf.concat((x2, d3), axis=4)
        d3 = self.up_conv_3(d3)

        d2 = self.up_2(d3)
        x1 = self.attention_2(g=d2, x=x1)
        d2 = tf.concat((x1, d2), axis=4)
        d2 = self.up_conv_2(d2)

        d1 = self.Conv_1x1(d2)
        output = d1[:, 15:161+15, 5:181+5, :, :]
        return output * self.mask

    def make(self, input_shape):
        '''
        This method makes the command "model.summary()" work.
        input_shape: (H,W,C), do not specify batch B
        '''
        x = tf.keras.layers.Input(shape=input_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name='actor')
        print(model.summary(line_length=120, show_trainable=True))
        return model

# -----------------------------------------------------------------------------------------------
class MyLSTMModel(Model):
    def __init__(self, prediction_length, mask):
        super().__init__()
        # input_shape = (batch_size, 10, 161, 181, features_amount)
        self.mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        self.days_prediction = prediction_length
        self.lstm = tf.keras.layers.ConvLSTM2D(filters=3*prediction_length, kernel_size=(3, 3), padding='same')
        # self.dense = tf.keras.layers.Dense(prediction_length)

    def call(self, x):
        # padding the image with zeroes to shape 192 x 192
        # x1 = tf.pad(x, [[0, 0], [15, 16], [5, 6], [0, 0], [0, 0]])
        x1 = self.lstm(x)
        # print(x1.shape)
        output = x1 * self.mask
        return tf.reshape(output, (-1, 161, 181, self.days_prediction, 3))

    def make(self, input_shape):
        '''
        This method makes the command "model.summary()" work.
        input_shape: (H,W,C), do not specify batch B
        '''
        x = tf.keras.layers.Input(shape=input_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x), name='actor')
        print(model.summary(line_length=120, show_trainable=True))
        return model