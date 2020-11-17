from keras import models
from keras import layers
import config as cfg
from tensorflow.keras import layers, models

class NN:
    def __init__(self, train, evaluation, test):
        self.train_input       = train[0]
        self.evaluation_input  = evaluation[0]
        self.test_input        = test[0]
        self.train_output      = train[1]
        self.evaluation_output = evaluation[1]
        self.test_output       = test[1]
        self.model = None


    def init_standard_convolutional_autoencoder(self):
        self.model = models.Sequential()
        input_shape = self.train_input[0].shape
        self.model.add(layers.Conv2D(8, (3, 3), padding="same", activation='relu', input_shape=input_shape))
        self.model.add(layers.MaxPooling2D((2, 2), padding="valid"))

        self.model.add(layers.Conv2D(12, (3, 3), padding="same", activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="valid"))

        self.model.add(layers.Conv2D(16, (3, 3), padding="same", activation='relu'))

        self.model.add(layers.UpSampling2D((2, 2), interpolation='bilinear', ))
        self.model.add(layers.Conv2D(12, (3, 3), padding="same", activation='relu'))

        self.model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
        self.model.add(layers.Conv2D(3, (3, 3), padding="same", activation='relu'))

        self.model.summary()
        self.model.compile(
            optimizer=cfg.nn["optimizer"],
            loss=cfg.nn["loss"],
            metrics=cfg.nn["metrics"]
        )


    def init_smart_convolutional_autoencoder(self):
        self.model = models.Sequential()
        input_shape = self.train_input[0].shape
        self.model.add(layers.Conv2D(8, (3, 3), padding="same", activation='relu', input_shape=input_shape))
        self.model.add(layers.MaxPooling2D((2, 2), padding="valid"))

        self.model.add(layers.Conv2D(16, (5, 5), padding="same", activation='relu'))
        self.model.add(layers.Conv2D(32, (3, 3), padding="same", activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="valid"))

        self.model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))

        self.model.add(layers.UpSampling2D((2, 2), interpolation='bilinear', ))
        self.model.add(layers.Conv2DTranspose(32, (3, 3), padding="same", activation='relu'))

        self.model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
        self.model.add(layers.Conv2DTranspose(3, (3, 3), padding="same", activation='relu'))

        self.model.summary()
        self.model.compile(
            optimizer=cfg.nn["optimizer"],
            loss=cfg.nn["loss"],
            metrics=cfg.nn["metrics"]
        )

    def init_colorizer_convolutional_autoencoder(self):
        self.model = models.Sequential()
        input_shape = self.train_input[0].shape
        self.model.add(layers.Conv2D(8, (3, 3), padding="same", activation='relu', input_shape=input_shape))
        self.model.add(layers.MaxPooling2D((2, 2), padding="valid"))

        self.model.add(layers.Conv2D(16, (5, 5), padding="same", activation='relu'))
        self.model.add(layers.Conv2D(32, (3, 3), padding="same", activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2), padding="valid"))

        self.model.add(layers.Conv2D(64, (3, 3), padding="same", activation='relu'))

        self.model.add(layers.UpSampling2D((2, 2), interpolation='bilinear', ))
        self.model.add(layers.Conv2DTranspose(32, (3, 3), padding="same", activation='relu'))

        self.model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
        self.model.add(layers.Conv2DTranspose(2, (3, 3), padding="same", activation='relu'))

        self.model.summary()
        self.model.compile(
            optimizer=cfg.nn["optimizer"],
            loss=cfg.nn["loss"],
            metrics=cfg.nn["metrics"]
        )


    def train_model(self, saving_path):
        # launch cnn

        hist = self.model.fit(
                        self.train_input,
                        self.train_output,
                        batch_size=cfg.nn["batch_size"],
                        epochs=cfg.nn["epochs"],
                        validation_data=(self.evaluation_input, self.evaluation_output)
                )

        # Calling `save('my_model.h5')` creates a h5 file `my_model.h5`.
        if saving_path:
            self.model.save(saving_path)

        # It can be used to reconstruct the model identically.
        #reconstructed_model = keras.models.load_model("my_h5_model.h5")

        return hist, self.model.layers


    def predict_output(self, input):
        return self.model.predict(input)


    def evaluate_model(self ):
        return self.model.evaluate(
            self.test_input,
            self.test_output
            )
