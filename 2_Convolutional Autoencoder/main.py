import tensorflow as tf
import pandas as pd
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import config as cfg
import nn
import time
import janitor as jn


# just to check if GPU is working
if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")


def extract_data():
    dataset = tf.keras.datasets.cifar10.load_data()
    dataset = np.vstack((dataset[0][0], dataset[1][0]))  # merged train and test set
    dataset = dataset / 255
    # dataset2 = np.random.shuffle(dataset)
    size = len(dataset)
    # recreate another set of (train, validation, test)
    return (dataset[:int(size * 0.8)], dataset[int(size * 0.8):int(size * 0.9)], dataset[int(size * 0.9):])


def plot_hist(history):
    f, ax = plt.subplots(1, 1) #, figsize=(10, 8))
    print(history.history)

    #summarize history for accuracy
    # ax[0].plot(history.history['accuracy'], c='C2')
    # ax[0].plot(history.history['val_accuracy'], c='C3')
    # ax[0].set_title('Model accuracy')
    # ax[0].set_ylabel('Accuracy')
    # ax[0].set_xlabel('Epoch')
    # ax[0].legend(['Train', 'Test'], loc='upper left')

    # summarize history for loss
    ax.plot(history.history['loss'], c='C0')
    ax.plot(history.history['val_loss'], c='C1')
    ax.set_title('Model loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')


def main():
    start = time.time()
    pickle_dataset_path = cfg.general["pickle_data_path"] / "dataset.pickle"
    dataset = jn.pickle_load(pickle_dataset_path)
    if not dataset:
        jn.create_dir(cfg.general["pickle_path"])
        dataset = extract_data()
        jn.pickle_save(dataset, pickle_dataset_path)

    print("Extraction and preprocessing time: ", str(time.time() - start))

    start = time.time()
    # TODO train

    cnn = nn.NN(dataset[0], dataset[1], dataset[2])

    cnn.init_convolutional_autoencoder()

    hist,layers = cnn.train_model()
    for layer in layers:
        print(layer.output_shape)
    plot_hist(hist)
    plt.savefig("img/plot_hist.png")
    plt.show()

    history = cnn.test_model()

    #print("Trained cnn -> ", cnn["id"])
    #print("CNN training time: ", str(time.time() - start))
    #print("public test acc  -> ", test_accuracy)
    #print("public test loss -> ", test_loss)

    # launch a list of cnn
    #cnn.cnn_P(training_data, private_test_data, public_test_data)

if __name__ == '__main__':
    main()