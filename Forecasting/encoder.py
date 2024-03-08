import tensorflow as tf
import keras

from Forecasting.nn_utilities import positional_encoding, feed_forward_network
from Forecasting.multi_head_attention import MultiHeadAttention


class EncoderLayer(keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, filter_size):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, filter_size)
        self.ffn = feed_forward_network(dff, d_model, filter_size)

        self.layernorm1 = keras.layers.BatchNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.BatchNormalization(epsilon=1e-6)

        # No dropouts for now

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        # (batch_size, input_seq_len, rows, cols, d_model)
        out1 = self.layernorm1(x + attn_output, training=training)
        # (batch_size, input_seq_len, rows, cols, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, rows, cols, d_model)
        out2 = self.layernorm2(out1 + ffn_output, training=training)

        return out2


class Encoder(keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, filter_size,
                 image_shape, max_position_encoding):
        super(Encoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = keras.layers.Conv2D(d_model, filter_size,
                                                padding='same', activation='relu')
        self.pos_encoding = positional_encoding(max_position_encoding, image_shape, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, filter_size)
                           for _ in range(num_layers)]

    def call(self, x, training, mask):
        # x.shape = (batch_size, seq_len, rows, cols, depth)
        seq_len = x.shape[1]

        # image embedding and position encoding
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :, :, :]
        # Incompatible
        # shapes: [4, 14, 161, 10]
        # vs.[1, 14, 161, 181, 10]

        for layer in range(self.num_layers):
            x = self.enc_layers[layer](x, training, mask)

        return x  # (batch_size, seq_len, rows, cols, d_model)