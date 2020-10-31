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

class NN:

    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.input_size  = n_inputs
        self.hidden_size = n_hidden
        self.output_size = n_outputs
        self.weights_0L  = np.random.randn(n_hidden, n_inputs)
        self.weights_1L  = np.random.randn(n_outputs, n_hidden)
        self.bias_0L     = np.ones((n_hidden, 1))
        self.bias_1L     = np.ones((n_outputs, 1))


    def __sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


    def __relu(self, x):
        return max(x, 0)


    def __derivative_sigmoid(self, x):
        return (np.exp(-x)) / ((np.exp(-x) + 1.0) ** 2)


    def __derivate_relu(self, x):
        return 1 if x > 0 else 0


    def __feed_forward(self, input):
        input = np.array([input]).T
        # sigm( (W0 * I) + b0 ) = H
        hidden_layer = self.__sigmoid(np.dot(self.weights_0L, input) + self.bias_0L)
        # sigm( (W1 * H) + b1 ) = O
        output = self.__sigmoid(np.dot(self.weights_1L, hidden_layer) + self.bias_1L)
        return output.T[0]

    def __back_propagation(self, obtained_output, expected_output):
        diff_vec = obtained_output - expected_output
        cost = 0
        for val_d in diff_vec:
            cost += diff_vec**2
        cost = 0.5*cost
        print(cost)
        derivative_cost = diff_vec * self.__derivative_sigmoid(obtained_output)




        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        #d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        #d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        #self.weights1 += d_weights1
        #self.weights2 += d_weights2

        return 1

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            for input in dataset:
                output = self.__feed_forward(input)
                asd = self.__back_propagation(output, input)


def main():
    neural_network = NN(INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER)

    mean_time = 0
    #for i in range(10):
    start_time = datetime.now()
    neural_network.train(DATASET, EPOCHS)
    end_time = datetime.now() - start_time
    mean_time += end_time.total_seconds()

    #print("Time spent: " + str(mean_time/10))


if __name__ == '__main__':
    main()
