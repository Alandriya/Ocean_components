import os
import tensorflow as tf
import keras
import math

# print("TensorFlow version:", tf.__version__)

class Transformer:
    def __init__(self, batch_size, n_days_lags, height, width):
        # input is a batch of batch_size videos of 2D frames 161х181, with n_days_lags frames per video and 3 channels:
        # X, A, B
        input_shape = (batch_size, 3, n_days_lags, height, width, 1)  # (4, 3, 7, 161, 181, 1)
        print(input_shape)
        x = tf.random.normal(input_shape)

        conv1 = keras.layers.Conv2D(filters=2, kernel_size=3, activation='relu', input_shape=input_shape[3:])
        pool1 = keras.layers.MaxPooling2D(2, input_shape=input_shape[3:])
        # conv2 = keras.layers.Conv3D(filters=16, kernel_size=3, activation='relu', input_shape=input_shape[2:])
        # pool2 = keras.layers.MaxPooling3D(2)
        # flat1 = keras.layers.Flatten()

        self.model = keras.models.Sequential([conv1, pool1])

        print(self.model.summary())
        print(self.model(x).shape)
        # model.compile(optimizer='adam',
        #               loss= keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #               metrics=[keras.metrics.SparseCategoricalAccuracy()])

    def save(self, checkpoint_path, epoch):
        self.model.save_weights(checkpoint_path + f"cp-{epoch:04d}.ckpt")

    def load(self, checkpoint_path, epoch=0):
        if epoch:
            self.model.load_weights(checkpoint_path + f"cp-{epoch:04d}.ckpt")
        else:
            latest = tf.train.latest_checkpoint(checkpoint_path)
            self.model.load_weights(latest)

    def train(self, checkpoint_path, X_train, Y_train, batch_size, epochs, mode='first'):
        n_batches = X_train.shape[0] / batch_size
        n_batches = math.ceil(n_batches)  # round up the number of batches to the nearest whole integer

        if mode != 'first':
            latest = tf.train.latest_checkpoint(checkpoint_path)
            self.model.load_weights(latest)
        else:
            if not os.path.exists(checkpoint_path):
                os.mkdir(checkpoint_path)

        # Create a callback that saves the model's weights every 5 epochs
        cp_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            save_freq=5 * n_batches)

        self.model.fit(X_train,
                  Y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=[cp_callback],
                  # validation_data=(test_images, test_labels),
                  verbose=0)

    def evaluate(self, X_test, Y_test):
        loss, acc = self.model.evaluate(X_test, Y_test, verbose=2)
        return loss, acc