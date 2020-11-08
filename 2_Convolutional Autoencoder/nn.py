from keras import models
from keras import layers
import config as cfg
import time
import janitor as jn
import tensorflow as tf
from keras.utils import to_categorical

from tensorflow.keras import datasets, layers, models

class NN:
    def __init__(self, train, evaluation, test):
        self.train_dataset      = train
        self.evaluation_dataset = evaluation
        self.test_dataset       = test
        self.model = None

    def init_convolutional_autoencoder(self):

        self.model = models.Sequential()
        input_shape = self.train_dataset[0].shape
        self.model.add(layers.Conv2D(8, (3, 3), padding="same", activation='relu', input_shape=input_shape))
        self.model.add(layers.MaxPooling2D((2, 2)))
#
        self.model.add(layers.Conv2D(12, (3, 3), padding="same", activation='relu'))
        self.model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#
#
#
        self.model.add(layers.Conv2D(16, (3, 3), padding="same", activation='relu'))
#
        self.model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
        self.model.add(layers.Conv2D(12, (3, 3), padding="same", activation='relu'))
#
        self.model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
        self.model.add(layers.Conv2D(3, (3, 3), padding="same", activation='relu'))

        #self.model.add(layers.Conv2D(8, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same',
        #                 input_shape=input_shape))
        #self.model.add(layers.MaxPooling2D((2, 2), padding='same'))
        #self.model.add(layers.Conv2D(8, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
        #self.model.add(layers.MaxPooling2D((2, 2), padding='same'))
        #self.model.add(layers.Conv2D(8, (2, 2), strides=(2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
        #self.model.add(layers.Conv2D(8, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
        #self.model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
        #self.model.add(layers.Conv2D(8, (2, 2), activation='relu'))
        #self.model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
        #self.model.add(layers.Conv2D(8, (2, 2), activation='relu', kernel_initializer='he_uniform', padding='same'))
        #self.model.add(layers.UpSampling2D((2, 2), interpolation='bilinear'))
        #self.model.add(layers.Conv2D(1, (2, 2), activation='sigmoid', padding='same'))

        #self.model.add(layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
        #self.model.add(layers.Conv2D(256, (3, 3), padding="same", activation='relu'))
        #self.model.add(layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
        #self.model.add(layers.Flatten())
#
        #self.model.add(layers.Dense(512, activation='relu'))
        #self.model.add(layers.Dropout(0.5))
        #self.model.add(layers.Dense(512, activation='relu'))
        #self.model.add(layers.Dropout(0.5))
        ## model.add(layers.Dense(2048, activation='relu'))
        ## model.add(layers.Dropout(0.2))
        #self.model.add(layers.Dense(7, activation='softmax'))
#
        #self.model.add(layers.Conv2D(32, (3, 3), padding="same", activation='relu', ))

        self.model.summary()


    def train_model(self):
        # launch cnn
        self.model.compile(
            optimizer=cfg.nn["optimizer"],
            loss=cfg.nn["loss"],
            metrics=cfg.nn["metrics"]
        )

        hist = self.model.fit(
                        self.train_dataset,
                        self.train_dataset,
                        batch_size=cfg.nn["batch_size"],
                        epochs=cfg.nn["epochs"],
                        validation_data=(self.evaluation_dataset, self.evaluation_dataset)
                )
        return hist

    def test_model(self):
        return self.model.evaluate(
            self.test_dataset,
            self.test_dataset
            )

    #jn.create_dir(config.general["pickle_history_path"])
    #saving_history_path = config.general["pickle_history_path"] / str("history_" + cnn["id"] + ".pickle")
    #history = {
    #    "id": cnn["id"],
    #    "epochs": cnn["epochs"],
    #    "history": hist.history,
    #    "test": [test_accuracy, test_loss]
    #}
    #jn.pickle_save(history, saving_history_path)
