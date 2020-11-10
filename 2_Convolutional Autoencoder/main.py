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
import cv2

TEST_STANDARD = True
TEST_SMART = True
TEST_COLORIZER = True

# just to check if GPU is working
if tf.test.gpu_device_name():
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")


def extract_data():
    dataset_rgb = tf.keras.datasets.cifar10.load_data()
    dataset_rgb = np.vstack((dataset_rgb[0][0], dataset_rgb[1][0]))  # merged train and test set

    #dataset_grayscale = np.dot(dataset_rgb[..., :3], [0.2989, 0.5870, 0.1140]) # https://e2eml.school/convert_rgb_to_grayscale.html
    #dataset_grayscale = np.around(dataset_grayscale) # better round numbers

    dataset_yuv = np.array([ cv2.cvtColor(img, cv2.COLOR_RGB2YUV) for img in dataset_rgb])

    # normalize image values from 0-255 to 0-1
    dataset_rgb_norm = dataset_rgb / 255.
    dataset_yuv_norm = dataset_yuv / 255.

    # recreate another set of (train, validation, test)
    size = len(dataset_rgb)
    return [
        (dataset_rgb[:int(size * 0.8)], dataset_rgb[int(size * 0.8):int(size * 0.9)], dataset_rgb[int(size * 0.9):]),
        (dataset_rgb_norm[:int(size * 0.8)], dataset_rgb_norm[int(size * 0.8):int(size * 0.9)], dataset_rgb_norm[int(size * 0.9):]),
        (dataset_yuv[:int(size * 0.8)], dataset_yuv[int(size * 0.8):int(size * 0.9)], dataset_yuv[int(size * 0.9):]),
        (dataset_yuv_norm[:int(size * 0.8)], dataset_yuv_norm[int(size * 0.8):int(size * 0.9)], dataset_yuv_norm[int(size * 0.9):])
    ]


def plot_hist_accuracy(history, img_name):
    f, ax = plt.subplots(1, 1)

    # summarize history for accuracy
    ax.plot(history.history['accuracy'], c='C0')
    ax.plot(history.history['val_accuracy'], c='C1')
    ax.set_title('Model accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(str(cfg.general["imgs_path"] / img_name))
    plt.show()


def plot_hist_loss(history, img_name):
    f, ax = plt.subplots(1, 1)

    # summarize history for loss
    ax.plot(history.history['loss'], c='C0')
    ax.plot(history.history['val_loss'], c='C1')
    ax.set_title('Model loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(str(cfg.general["imgs_path"] / img_name))
    plt.show()


def plot_images_resulting(cnn, dataset, img_name):
    predicted_dataset = cnn.predict_output(dataset)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1600, 1600)

    index = [2, 3, 5, 6]
    # reconstruct image
    img = np.float32(dataset[0])# np.around(dataset[0]*255)
    #img = np.float32(img)
    pred_img = predicted_dataset[0]#np.around(predicted_dataset[0]*255)
    #pred_img = np.float32(pred_img)

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predicted_image = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

    result = np.vstack([np.hstack([image, predicted_image])])

    for i in index:
        img = np.float32(dataset[i])
        pred_img = predicted_dataset[i]
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predicted_image = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
        result = np.vstack([result, np.hstack([image, predicted_image])])


    cv2.imshow('image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    #result = np.around(result*255)
    #result = np.array(result, np.int)
    ##image.convertTo(result, cv2.CV_8UC3, 255.0)
    #cv2.imwrite("asd.png", result)

    #n = 5
    #plt.figure(figsize=(32, 4))
    #for i in range(1, n + 1):
    #    # Display original
    #    ax = plt.subplot(2, n, i)
    #    plt.imshow(dataset[i])
    #    plt.gray()
    #    ax.get_xaxis().set_visible(False)
    #    ax.get_yaxis().set_visible(False)

    #    # Display reconstruction
    #    ax = plt.subplot(2, n, i + n)
    #    plt.imshow(predicted_dataset[i])
    #    plt.gray()
    #    ax.get_xaxis().set_visible(False)
    #    ax.get_yaxis().set_visible(False)
    #plt.savefig(str(cfg.general["imgs_path"] / img_name))
    #plt.show()


def plot_colorized_images_resulting(cnn, dataset, img_name):
    dataset_y = np.expand_dims(dataset[:, :, :, 0], axis=3)
    predicted_uv = cnn.predict_output(dataset_y)


    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1600, 1600)

    index = [2, 3, 5, 6]

    # reconstruct image
    img = np.float32(dataset[0])# np.around(dataset[0]*255)

    y = np.squeeze(np.float32(dataset_y[0]))
    u, v = cv2.split(predicted_uv[0])
    pred_img = cv2.merge((y, u, v))

    image = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
    predicted_image = cv2.cvtColor(pred_img, cv2.COLOR_YUV2BGR)

    result = np.vstack([np.hstack([image, predicted_image])])

    for i in index:
        img = np.float32(dataset[i])  # np.around(dataset[0]*255)

        y = np.squeeze(np.float32(dataset_y[i]))
        u, v = cv2.split(predicted_uv[i])
        pred_img = cv2.merge((y, u, v))

        image = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        predicted_image = cv2.cvtColor(pred_img, cv2.COLOR_YUV2BGR)
        result = np.vstack([result, np.hstack([image, predicted_image])])


    cv2.imshow('image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    jn.create_dir(cfg.general["imgs_path"])
    jn.create_dir(cfg.general["models_path"])

    start = time.time()
    pickle_dataset_path = cfg.general["pickle_path"] / "dataset.pickle"
    dataset_list = jn.pickle_load(pickle_dataset_path)
    if not dataset_list:
        jn.create_dir(cfg.general["pickle_path"])
        dataset_list = extract_data()
        jn.pickle_save(dataset_list, pickle_dataset_path)

    print("\nExtraction and preprocessing time: {:0.3f} sec\n".format(time.time() - start))

    dataset_rgb, dataset_rgb_norm, dataset_yuv, dataset_yuv_norm = dataset_list


    ####################################################################################################################
    if TEST_STANDARD:
        train_dataset, evaluation_dataset, test_dataset = dataset_rgb_norm[0], dataset_rgb_norm[1], dataset_rgb_norm[2]

        cnn_standard = nn.NN((train_dataset, train_dataset), (evaluation_dataset, evaluation_dataset), (test_dataset, test_dataset))
        cnn_standard.init_standard_convolutional_autoencoder()

        start = time.time()
        hist_standard, layers_standard = cnn_standard.train_model(str(cfg.general["models_path"] / "standard_model.h5"))
        print("\nCNN standard training time: {:0.3f} sec\n".format(time.time() - start))


        plot_hist_accuracy(hist_standard, "accuracy_standard.png")
        plot_hist_loss(hist_standard, "loss_standard.png")
        plot_images_resulting(cnn_standard, test_dataset, "confront_standard.png")

        hist_standard_path = cfg.general["pickle_path"] / "hist_standard.pickle"
        jn.pickle_save(hist_standard.history, hist_standard_path)

        test_loss_standard, test_accuracy_standard = cnn_standard.evaluate_model()
        print("Test Accuracy standard -> ", test_loss_standard)
        print("Test Loss standard -> ", test_accuracy_standard)
    ####################################################################################################################


    ####################################################################################################################
    if TEST_SMART:
        cnn_smart = nn.NN((train_dataset, train_dataset), (evaluation_dataset, evaluation_dataset), (test_dataset, test_dataset))
        cnn_smart.init_smart_convolutional_autoencoder()

        start = time.time()
        hist_smart, layers_smart = cnn_smart.train_model(str(cfg.general["models_path"] / "smart_model.h5"))
        print("\nCNN smart training time: {:0.3f} sec\n".format(time.time() - start))

        plot_hist_accuracy(hist_smart, "accuracy_smart.png")
        plot_hist_loss(hist_smart, "loss_smart.png")
        plot_images_resulting(cnn_smart, test_dataset, "confront_smart.png")

        hist_smart_path = cfg.general["pickle_path"] / "hist_smart.pickle"
        jn.pickle_save(hist_smart.history, hist_smart_path)

        test_loss_smart, test_accuracy_smart = cnn_smart.evaluate_model()
        print("Test Accuracy smart -> ", test_loss_smart)
        print("Test Loss smart -> ", test_accuracy_smart)
    ####################################################################################################################


    ####################################################################################################################
    if TEST_COLORIZER:
        train_dataset, evaluation_dataset, test_dataset = dataset_yuv_norm[0], dataset_yuv_norm[1], dataset_yuv_norm[2]
        train_input, evaluation_input, test_input    = np.expand_dims(train_dataset[:, :, :, 0], axis=3), np.expand_dims(evaluation_dataset[:, :, :, 0], axis=3), np.expand_dims(test_dataset[:, :, :, 0], axis=3)
        train_output, evaluation_output, test_output = train_dataset[:, :, :, 1:], evaluation_dataset[:, :, :, 1:], test_dataset[:, :, :, 1:]

        cnn_colorizer = nn.NN((train_input, train_output), (evaluation_input, evaluation_output), (test_input, test_output))
        cnn_colorizer.init_colorizer_convolutional_autoencoder()

        start = time.time()
        hist_colorizer, layers_colorizer = cnn_colorizer.train_model(str(cfg.general["models_path"] / "colorizer_model.h5"))
        print("\nCNN colorizer training time: {:0.3f} sec\n".format(time.time() - start))

        plot_hist_accuracy(hist_colorizer, "accuracy_colorizer.png")
        plot_hist_loss(hist_colorizer, "loss_colorizer.png")
        plot_colorized_images_resulting(cnn_colorizer, test_dataset, "confront_colorizer.png")

        hist_colorizer_path = cfg.general["pickle_path"] / "hist_colorizer.pickle"
        jn.pickle_save(hist_colorizer.history, hist_colorizer_path)

        test_loss_colorizer, test_accuracy_colorizer = cnn_colorizer.evaluate_model()
        print("Test Accuracy colorizer -> ", test_loss_colorizer)
        print("Test Loss colorizer -> ", test_accuracy_colorizer)
    ####################################################################################################################


 
if __name__ == '__main__':
    main()