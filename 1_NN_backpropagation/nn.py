import numpy as np
from datetime import datetime
from scipy.special import expit
import time



INPUT_LAYER = 8
HIDDEN_LAYER = 3
OUTPUT_LAYER = 8
EPOCHS = 1000

#DATASET = [
#    np.array([1, 0, 0, 0, 0, 0, 0, 0]),
#    np.array([0, 1, 0, 0, 0, 0, 0, 0]),
#    np.array([0, 0, 1, 0, 0, 0, 0, 0]),
#    np.array([0, 0, 0, 1, 0, 0, 0, 0]),
#    np.array([0, 0, 0, 0, 1, 0, 0, 0]),
#    np.array([0, 0, 0, 0, 0, 1, 0, 0]),
#    np.array([0, 0, 0, 0, 0, 0, 1, 0]),
#    np.array([0, 0, 0, 0, 0, 0, 0, 1])
#]
DATASET = np.identity(8)
#DATASET = np.array([1, 0, 0, 0, 0, 0, 0, 0]).T
LABEL = DATASET

class NN:

    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.input_size  = n_inputs
        self.hidden_size = n_hidden
        self.output_size = n_outputs
        self.W1  = np.random.randn(n_hidden, n_inputs)
        self.W2  = np.random.randn(n_outputs, n_hidden)
        self.B1    = np.ones((n_hidden, 1))
        self.B2     = np.ones((n_outputs, 1))
        self.A1 = None
        self.A2 = None


    def __sigmoid(self, z):
        return expit(z)
        # return 1.0 / (1.0 + np.exp(-z))


    def __derivative_sigmoid(self, z):
        return self.__sigmoid(z) * (1-self.__sigmoid(z))


    def __relu(self, z):
        return max(z, 0)


    def __derivate_relu(self, z):
        return 1 if z > 0 else 0


    def __forward_propagation(self, input):
        #input = np.array([input]).T
        # sigm( (W1 * I) + B1 ) = A1
        Z1 = np.dot(self.W1, input) + self.B1
        self.A1 = self.__sigmoid(Z1)
        # sigm( (W2 * A1) + B2 ) = A2
        Z2 = np.dot(self.W2, self.A1) + self.B2
        self.A2 = self.__sigmoid(Z2)
        return self.A2

    def __cost_function(self, Y_expected):
        mean_cost = 0
        diff_vec_Y = (self.A2 - Y_expected).T
        for val_d in diff_vec_Y:
            cost = 0
            for val in val_d:
                cost += val**2
            mean_cost += 0.5*cost
        return (1/8) * mean_cost


    #def __back_propagation(self, Y, Y_pred):
    #    diff_vec_Y = Y - Y_pred

    #    derivative_cost = diff_vec_Y * self.__derivative_sigmoid(Y)
    #    diff_matrix_1L = np.dot(self.W1.T, derivative_cost)
    #    diff_matrix_0L = np.dot(self.input.T,  (np.dot(derivative_cost, self.W2.T) * self.__derivative_sigmoid(self.layer1)))

    def __back_propagation(self, X, Y_expected, learning_rate):
        #(dataset, output, 1)
        # Backward propagation: calculate dW1, db1, dW2, db2.
        d_Z2 = self.A2 - Y_expected
        d_W2 = (1 / 8) * np.dot(d_Z2, self.A1.T)
        d_B2 = (1 / 8) * np.sum(d_Z2, axis=1, keepdims=True)

        size_d_Z2 = d_Z2.shape
        size_d_W2 = d_W2.shape
        size_d_B2 = d_B2.shape

        size_Z2 = self.A2.shape
        size_W2 = self.W2.shape
        size_B2 = self.B2.shape

        d_Z1 = np.multiply(np.dot(self.W2.T, d_Z2), 1 - np.power(self.A1, 2))
        size_d_Z1 = d_Z1.shape
        size_Z1 = self.A1.shape
        d_W1 = (1 / 8) * np.dot(d_Z1, X.T)
        size_d_W1 = d_W1.shape
        size_W1 = self.W1.shape
        d_B1 = (1 / 8) * np.sum(d_Z1, axis=1, keepdims=True)



        size_d_Z1 = d_Z1.shape
        size_d_W1 = d_W1.shape
        size_d_B1 = d_B1.shape

        size_Z1 = self.A1.shape
        size_W1 = self.W1.shape
        size_B1 = self.B1.shape

        # Updating the parameters according to algorithm
        self.W1 = self.W1 - learning_rate * d_W1
        self.B1 = self.B1 - learning_rate * d_B1
        self.W2 = self.W2 - learning_rate * d_W2
        self.B2 = self.B2 - learning_rate * d_B2


    def train(self, dataset, label, epochs):
        for epoch in range(epochs):

            output = self.__forward_propagation(dataset)
            print(self.__cost_function(label))
            asd = self.__back_propagation(dataset, label, 1)


def main():
    neural_network = NN(INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER)

    mean_time = 0
    #for i in range(10):
    start_time = datetime.now()
    neural_network.train(DATASET, LABEL, EPOCHS)
    end_time = datetime.now() - start_time
    mean_time += end_time.total_seconds()

    #print("Time spent: " + str(mean_time/10))


if __name__ == '__main__':
    main()
