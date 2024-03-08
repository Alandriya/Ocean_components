


# print("TensorFlow version:", tf.__version__)

# class Transformer:
#     def __init__(self, batch_size, n_days_lags, height, width):
#         # input is a batch of batch_size videos of 2D frames 161х181, with n_days_lags frames per video and 3 channels:
#         # X, A, B
#         input_shape = (batch_size, 3, n_days_lags, height, width, 1)  # (4, 3, 7, 161, 181, 1)
#         print(input_shape)
#         x = tf.random.normal(input_shape)
#
#         conv1 = keras.layers.Conv2D(filters=2, kernel_size=3, activation='relu', input_shape=input_shape[3:])
#         pool1 = keras.layers.MaxPooling2D(2, input_shape=input_shape[3:])
#         # conv2 = keras.layers.Conv3D(filters=16, kerne1l_size=3, activation='relu', input_shape=input_shape[2:])
#         # pool2 = keras.layers.MaxPooling3D(2)
#         # flat1 = keras.layers.Flatten()
#
#         self.model = keras.models.Sequential([conv1, pool1])
#
#         print(self.model.summary())
#         print(self.model(x).shape)
#         # model.compile(optimizer='adam',
#         #               loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#         #               metrics=[keras.metrics.SparseCategoricalAccuracy()])
#
#     def save(self, checkpoint_path, epoch):
#         self.model.save_weights(checkpoint_path + f"cp-{epoch:04d}.ckpt")
#
#     def load(self, checkpoint_path, epoch=0):
#         if epoch:
#             self.model.load_weights(checkpoint_path + f"cp-{epoch:04d}.ckpt")
#         else:
#             latest = tf.train.latest_checkpoint(checkpoint_path)
#             self.model.load_weights(latest)
#
#     def train(self, checkpoint_path, X_train, Y_train, batch_size, epochs, mode='first'):
#         n_batches = X_train.shape[0] / batch_size
#         n_batches = math.ceil(n_batches)  # round up the number of batches to the nearest whole integer
#
#         if mode != 'first':
#             latest = tf.train.latest_checkpoint(checkpoint_path)
#             self.model.load_weights(latest)
#         else:
#             if not os.path.exists(checkpoint_path):
#                 os.mkdir(checkpoint_path)
#
#         # Create a callback that saves the model's weights every 5 epochs
#         cp_callback = keras.callbacks.ModelCheckpoint(
#             filepath=checkpoint_path,
#             verbose=1,
#             save_weights_only=True,
#             save_freq=5 * n_batches)
#
#         self.model.fit(X_train,
#                   Y_train,
#                   epochs=epochs,
#                   batch_size=batch_size,
#                   callbacks=[cp_callback],
#                   # validation_data=(test_images, test_labels),
#                   verbose=0)
#
#     def evaluate(self, X_test, Y_test):
#         loss, acc = self.model.evaluate(X_test, Y_test, verbose=2)
#         return loss, acc




import tensorflow as tf
import keras
import time
import numpy as np
from Forecasting.transformer import Transformer
from Forecasting.nn_utilities import create_look_ahead_mask


class VideoPrediction:
    def __init__(self, num_layers, d_model, num_heads, dff, filter_size, image_shape,
                 pe_input, pe_target, out_channel, loss_function='mse', optimizer='rmsprop'):
        self.transformer = Transformer(num_layers, d_model, num_heads, dff, filter_size,
                                       image_shape, pe_input, pe_target, out_channel)
        self.loss_object = keras.losses.MeanSquaredError() \
            if loss_function == 'mse' else keras.losses.BinaryCrossentropy()
        self.optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9) \
            if optimizer == 'rmsprop' else keras.optimizers.Adam()

    def loss_function(self, real, pred):
        return self.loss_object(real, pred)

    def train_step(self, inp, tar):

        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        look_ahead_mask = create_look_ahead_mask(tar.shape[1])
        loss = 0

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer(inp, tar_inp, True, look_ahead_mask)
            loss = self.loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        return loss

    def train(self, inp, tar, epochs, batch_size, checkpoint_path, epoch_print=5):

        start = time.time()
        for epoch in range(epochs):
            total_loss = 0
            total_batch = inp.shape[0] // batch_size

            for batch in range(total_batch):
                index = batch * batch_size
                enc_inp = inp[index:index + batch_size, :, :, :]
                dec_inp = tar[index:index + batch_size, :, :, :]

                batch_loss = self.train_step(enc_inp, dec_inp)
                total_loss += batch_loss

            total_batch += 1
            if epoch % epoch_print == 0:
                # Create a callback that saves the model's weights
                self.save(checkpoint_path + f'{epoch}')

                print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / total_batch))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
                start = time.time()

    def eval_step(self, inp, tar):

        batch_loss = 0
        image_size = inp.shape[2:]

        look_ahead_mask = create_look_ahead_mask(tar.shape[1])

        half = inp.shape[1] // 2
        output = inp[:, half:, :, :, :]
        encoder_input = inp[:, :half, :, :, :]

        for t in range(tar.shape[1]):
            prediction, _ = self.transformer(
                encoder_input, output, False, look_ahead_mask
            )

            predict = prediction[:, -1:, :, :, :]
            batch_loss += self.loss_function(tar[:, t:t + 1], predict)

            output = tf.concat([output, predict], axis=1)
            encoder_input = tf.concat([encoder_input, output[:, 0:1]], axis=1)[:, 1:]
            output = output[:, 1:]

        return (batch_loss / int(tar.shape[1]))

    def evaluate(self, inp, tar, batch_size):

        start = time.time()
        total_loss = 0
        total_batch = inp.shape[0] // batch_size

        for batch in range(total_batch):
            index = batch * batch_size
            enc_inp = inp[index:index + batch_size, :, :, :]
            dec_inp = tar[index:index + batch_size, :, :, :]

            batch_loss = self.eval_step(enc_inp, dec_inp)
            total_loss += batch_loss

        total_batch += 1

        return total_loss / total_batch

    def predict(self, inp, tar_seq_len):

        inp = tf.expand_dims(inp, 0)
        image_size = inp.shape[2:]

        look_ahead_mask = create_look_ahead_mask(tar_seq_len)

        predictions = []
        half = inp.shape[1] // 2
        output = inp[:, half:, :, :, :]
        encoder_input = inp[:, :half:, :, :]

        for t in range(tar_seq_len):
            prediction, _ = self.transformer(
                encoder_input, output, False, look_ahead_mask
            )

            predict = prediction[:, -1:, :, :, :]
            output = tf.concat([output, predict], axis=1)

            encoder_input = tf.concat([encoder_input, output[:, 0:1]], axis=1)[:, 1:]
            output = output[:, 1:]

            predictions.append(
                predict.numpy().reshape(
                    image_size
                )
            )

        return np.array(predictions)

    def save(self, path):
        self.transformer.save(path)
        pass

    def load(self, path):
        self.transformer.load_weights(path)
        pass