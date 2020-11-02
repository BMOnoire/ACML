import numpy as np
from datetime import datetime
import time



INPUT_LAYER = 8
HIDDEN_LAYER = 3
OUTPUT_LAYER = 8
EPOCHS = 10000

DATASET = [
    np.array([1, 0, 0, 0, 0, 0, 0, 0]),
    np.array([0, 1, 0, 0, 0, 0, 0, 0]),
    np.array([0, 0, 1, 0, 0, 0, 0, 0]),
    np.array([0, 0, 0, 1, 0, 0, 0, 0]),
    np.array([0, 0, 0, 0, 1, 0, 0, 0]),
    np.array([0, 0, 0, 0, 0, 1, 0, 0]),
    np.array([0, 0, 0, 0, 0, 0, 1, 0]),
    np.array([0, 0, 0, 0, 0, 0, 0, 1])
]
LABEL = [
    np.array([1, 0, 0, 0, 0, 0, 0, 0]),
    np.array([0, 1, 0, 0, 0, 0, 0, 0]),
    np.array([0, 0, 1, 0, 0, 0, 0, 0]),
    np.array([0, 0, 0, 1, 0, 0, 0, 0]),
    np.array([0, 0, 0, 0, 1, 0, 0, 0]),
    np.array([0, 0, 0, 0, 0, 1, 0, 0]),
    np.array([0, 0, 0, 0, 0, 0, 1, 0]),
    np.array([0, 0, 0, 0, 0, 0, 0, 1])
]

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
        return 1.0 / (1.0 + np.exp(-z))


    def __derivative_sigmoid(self, z):
        return self.__sigmoid(z) * (1-self.__sigmoid(z))


    def __relu(self, z):
        return max(z, 0)


    def __derivate_relu(self, z):
        return 1 if z > 0 else 0


    def __feed_forward(self, input):
        input = np.array([input]).T
        # sigm( (W1 * I) + B1 ) = A1
        Z1 = np.dot(self.W1, input) + self.B1
        self.A1 = self.__sigmoid(Z1)
        # sigm( (W2 * A1) + B2 ) = A2
        Z2 = np.dot(self.W2, self.A1) + self.B2
        self.A2 = self.__sigmoid(Z2)
        return self.A2.T[0]


    def __back_propagation(self, Y, Y_pred):
        diff_vec_Y = Y - Y_pred
        cost = 0
        for val_d in diff_vec_Y:
            cost += diff_vec_Y**2
        cost = 0.5*cost
        print(cost)
        derivative_cost = diff_vec_Y * self.__derivative_sigmoid(Y)
        diff_matrix_1L = np.dot(self.W1.T, derivative_cost)
        diff_matrix_0L = np.dot(self.input.T,  (np.dot(derivative_cost, self.W2.T) * self.__derivative_sigmoid(self.layer1)))

        def back_propagate(W1, b1, W2, b2, cache):
            # Retrieve also A1 and A2 from dictionary "cache"
            A1 = cache['A1']
            A2 = cache['A2']

            # Backward propagation: calculate dW1, db1, dW2, db2.
            dZ2 = A2 - Y
            dW2 = (1 / m) * np.dot(dZ2, A1.T)
            db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

            dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
            dW1 = (1 / m) * np.dot(dZ1, X.T)
            db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

            # Updating the parameters according to algorithm
            W1 = W1 - learning_rate * dW1
            b1 = b1 - learning_rate * db1
            W2 = W2 - learning_rate * dW2
            b2 = b2 - learning_rate * db2

            return W1, W2, b1, b2


            # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        #d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        #d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        #self.weights1 += d_weights1
        #self.weights2 += d_weights2

        return 1

    def train(self, dataset, label, epochs):
        for epoch in range(epochs):
            for input in dataset:
                output = self.__feed_forward(input)
                asd = self.__back_propagation(output, input)


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
