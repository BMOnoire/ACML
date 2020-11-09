import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # lighter system log

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
    dataset_rgb = tf.keras.datasets.cifar10.load_data()
    dataset_rgb = np.vstack((dataset_rgb[0][0], dataset_rgb[1][0]))  # merged train and test set

    dataset_grayscale = np.dot(dataset_rgb[..., :3], [0.2989, 0.5870, 0.1140]) # https://e2eml.school/convert_rgb_to_grayscale.html
    dataset_grayscale = np.around(dataset_grayscale) # better round numbers

    # normalize image values from 0-255 to 0-1
    dataset_rgb_norm = dataset_rgb / 255.
    dataset_grayscale_norm = dataset_grayscale / 255.

    # recreate another set of (train, validation, test)
    size = len(dataset_rgb)
    return [
        (dataset_rgb[:int(size * 0.8)], dataset_rgb[int(size * 0.8):int(size * 0.9)], dataset_rgb[int(size * 0.9):]),
        (dataset_rgb_norm[:int(size * 0.8)], dataset_rgb_norm[int(size * 0.8):int(size * 0.9)], dataset_rgb_norm[int(size * 0.9):]),
        (dataset_grayscale[:int(size * 0.8)], dataset_grayscale[int(size * 0.8):int(size * 0.9)], dataset_grayscale[int(size * 0.9):]),
        (dataset_grayscale_norm[:int(size * 0.8)], dataset_grayscale_norm[int(size * 0.8):int(size * 0.9)], dataset_grayscale_norm[int(size * 0.9):])
    ]


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
    plt.savefig(str(cfg.general["imgs_path"] / "plot_hist.png"))
    plt.show()


def plot_images_resulting(cnn, dataset):
    predicted_dataset = cnn.predict_output(dataset)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(dataset[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(predicted_dataset[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(str(cfg.general["imgs_path"] / "pred.png"))
    plt.show()

    # ax[i].set_title


def main():
    start = time.time()
    jn.create_dir(cfg.general["imgs_path"])
    pickle_dataset_path = cfg.general["pickle_path"] / "dataset.pickle"
    dataset_list = jn.pickle_load(pickle_dataset_path)
    if not dataset_list:
        jn.create_dir(cfg.general["pickle_path"])
        dataset_list = extract_data()
        jn.pickle_save(dataset_list, pickle_dataset_path)

    print("\nExtraction and preprocessing time: {:0.3f} sec\n".format(time.time() - start))

    dataset_rgb, dataset_rgb_norm, dataset_grayscale, dataset_grayscale_norm = dataset_list

    ####################################################################################################################
    train_dataset, evaluation_dataset, test_dataset = dataset_rgb_norm[0], dataset_rgb_norm[1], dataset_rgb_norm[2]

    cnn_standard = nn.NN(train_dataset, evaluation_dataset, test_dataset)
    cnn_standard.init_standard_convolutional_autoencoder()

    start = time.time()
    hist_standard, layers_standard = cnn_standard.train_model()
    print("\nCNN standard training time: {:0.3f} sec\n".format(time.time() - start))

    plot_hist(hist_standard)
    plot_images_resulting(cnn_standard, train_dataset)

    test_loss_standard, test_accuracy_standard = cnn_standard.evaluate_model()
    print("Test Accuracy standard -> ", test_loss_standard)
    print("Test Loss standard -> ", test_accuracy_standard)
    ####################################################################################################################

    ####################################################################################################################
    cnn_smart = nn.NN(train_dataset, evaluation_dataset, test_dataset)
    cnn_smart.init_smart_convolutional_autoencoder()

    start = time.time()
    hist_smart, layers_smart = cnn_smart.train_model()
    print("\nCNN smart training time: {:0.3f} sec\n".format(time.time() - start))

    plot_hist(hist_smart)
    plot_images_resulting(cnn_smart, train_dataset)

    test_loss_smart, test_accuracy_smart = cnn_smart.evaluate_model()
    print("Test Accuracy smart -> ", test_loss_smart)
    print("Test Loss smart -> ", test_accuracy_smart)
    ####################################################################################################################

    ####################################################################################################################
    # TODO gray
    ####################################################################################################################



if __name__ == '__main__':
    main()