import tensorflow as tf
import keras

from Forecasting.nn_utilities import positional_encoding, feed_forward_network
from Forecasting.multi_head_attention import MultiHeadAttention


class DecoderLayer(keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, filter_size):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads, filter_size)
        self.mha2 = MultiHeadAttention(d_model, num_heads, filter_size)

        self.ffn = feed_forward_network(dff, d_model, filter_size)

        self.layernorm1 = keras.layers.BatchNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.BatchNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.BatchNormalization(epsilon=1e-6)

        # No dropouts for now

    def call(self, x, enc_output, training, look_ahead_mask):
        # enc_output.shape = (batch_size, input_seq_len, rows, cols, d_model)
        # x.shape = (batch_size, input_seq_len, rows, cols, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        # (batch_size, target_seq_len, rows, cols, d_model)

        out1 = self.layernorm1(x + attn1, training=training)
        # (batch_size, target_seq_len, rows, cols, d_model)

        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, None)
        # (batch_size, target_seq_len, rows, cols, d_model)

        out2 = self.layernorm2(out1 + attn2, training=training)
        # (batch_size, target_seq_len, rows, cols, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, rows, cols, d_model)
        out3 = self.layernorm3(out2 + ffn_output, training=training)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, filter_size,
                 image_shape, max_position_encoding):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = keras.layers.Conv2D(d_model, filter_size,
                                                padding='same', activation='relu')
        self.pos_encoding = positional_encoding(max_position_encoding, image_shape, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, filter_size)
                           for _ in range(num_layers)]

    def call(self, x, enc_output, training, look_ahead_mask):
        # enc_output.shape = (batch_size, input_seq_len, rows, cols, depth)

        seq_len = x.shape[1]
        attention_weights = {}

        # image embedding and position encoding
        x = self.embedding(x)  # (batch_size, target_seq_len, rows, cols, depth)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :, :, :]

        for layer in range(self.num_layers):
            x, block1, block2 = self.dec_layers[layer](x, enc_output,
                                                       training, look_ahead_mask)

            attention_weights[f'decoder_layer{layer + 1}_block1'] = block1
            attention_weights[f'decoder_layer{layer + 1}_block2'] = block2

        # x.shape = (batch_size, target_seq_len, rows, cols, d_model)
        return x, attention_weights