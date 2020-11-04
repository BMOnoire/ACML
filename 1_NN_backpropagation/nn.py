import numpy as np
from datetime import datetime
from scipy.special import expit
import time
import matplotlib.pyplot as plt


INPUT_LAYER = 8
HIDDEN_LAYER = 3
OUTPUT_LAYER = 8
EPOCHS = 50000

#set the seed
np.random.seed(2)

DATASET = np.identity(8)
LABEL = DATASET
TEST = LABEL[:, 3:4]  # take already the column instead of transposing it
LEARNING_RATE = [0.01, 0.05, 0.1, 0.5, 1, 2, 3, 5, 10]


class NN:

    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.input_size  = n_inputs
        self.hidden_size = n_hidden
        self.output_size = n_outputs
        self.W1 = None
        self.W2 = None
        self.B1 = None
        self.B2 = None
        self.A1 = None
        self.A2 = None

    """initialize the parameters for the NN"""
    def init_random_nn(self):
        W1 = np.random.randn(self.hidden_size, self.input_size) * (0.01)
        W2 = np.random.randn(self.output_size, self.hidden_size) * (0.01)
        B1 = np.random.randn(self.hidden_size, 1) * (0.01)
        B2 = np.random.randn(self.output_size, 1) * (0.01)
        return W1, W2, B1, B2


    def add_weight_and_biases(self, W1, W2, B1, B2):
        self.W1 = W1
        self.W2 = W2
        self.B1 = B1
        self.B2 = B2


    def __sigmoid(self, z):
        return expit(z) # 1.0 / (1.0 + np.exp(-z))


    def __derivative_sigmoid(self, z):
        return self.__sigmoid(z) * (1-self.__sigmoid(z))


    def __forward_propagation(self, input):
        Z1 = np.dot(self.W1, input) + self.B1
        self.A1 = self.__sigmoid(Z1)
        Z2 = np.dot(self.W2, self.A1) + self.B2
        self.A2 = self.__sigmoid(Z2)


    def __cost_function(self, Y):
        m = Y.shape[1]
        cost_sum = np.multiply(np.log(self.A2), Y) + np.multiply((1 - Y), np.log(1 - self.A2))
        cost = - np.sum(cost_sum) / m
        return cost


    def __back_propagation(self, X, Y, learning_rate):
        m = X.shape[1]

        d_Z2 = self.A2 - Y
        d_Z1 = np.multiply(np.dot(self.W2.T, d_Z2), 1 - np.power(self.A1, 2))

        d_B2 = (1 / m) * np.sum(d_Z2, axis=1, keepdims=True)
        d_B1 = (1 / m) * np.sum(d_Z1, axis=1, keepdims=True)

        d_W2 = (1 / m) * np.dot(d_Z2, self.A1.T)
        d_W1 = (1 / m) * np.dot(d_Z1, X.T)

        # Updating the parameters according to the algorithm
        self.W1 = self.W1 - learning_rate * d_W1
        self.B1 = self.B1 - learning_rate * d_B1
        self.W2 = self.W2 - learning_rate * d_W2
        self.B2 = self.B2 - learning_rate * d_B2

    def train(self, dataset, label, epochs, learning_rate):
        #array to save the costs to be plotted
        cost_list = []
        for epoch in range(epochs):
            self.__forward_propagation(dataset)
            cost = self.__cost_function(label)
            cost_list.append([epoch,cost])
            self.__back_propagation(dataset, label, learning_rate)

        return cost_list

    #plot the array cost_list with the epochs
    def plot_cost_graph(self, value_list, name_list):
        for value in value_list:
            plt.plot(*zip(*value))
        plt.ylabel("cost")
        plt.xlabel("epoch")
        plt.legend(name_list, loc='upper right')
        plt.show()

    #perform the prediction of an input node to show the actual predicted results
    def test_prediction(self, x):

        Z1 = np.dot(self.W1, x) + self.B1
        self.A1 = self.__sigmoid(Z1)
        Z2 = np.dot(self.W2, self.A1) + self.B2
        output = self.__sigmoid(Z2)
        return output


def main():
    neural_network = NN(INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER)

    W1, W2, B1, B2 = neural_network.init_random_nn()
    mean_time = 0
    cost_plot_list = []

    print("\nThe input for the test is:")
    print(TEST)

    #perform the training for every different learnign rate
    for learning_rate in LEARNING_RATE:
        neural_network.add_weight_and_biases(W1, W2, B1, B2)

        start_time = datetime.now()
        cost_list = neural_network.train(DATASET, LABEL, EPOCHS, learning_rate)
        end_time = datetime.now() - start_time

        test_output = neural_network.test_prediction(TEST)
        print("\nThe output with learning rate " + str(learning_rate) + " is:")
        print(np.round(test_output, 3))

        cost_plot_list.append(cost_list)
        mean_time += end_time.total_seconds()

    print("\nTrained NN with", EPOCHS, "epochs")
    print("\nTraining completed in", round(mean_time/len(LEARNING_RATE),4)) # is the average of all the trainings with different LEARNING_RATE

    neural_network.plot_cost_graph(cost_plot_list, LEARNING_RATE)


if __name__ == '__main__':
    main()