import numpy as np
from datetime import datetime
from scipy.special import expit
import time
import matplotlib.pyplot as plt


INPUT_LAYER = 8
HIDDEN_LAYER = 3
OUTPUT_LAYER = 8
EPOCHS = 5000
DATASET = np.identity(8)
print(DATASET)
LABEL = DATASET

test = LABEL[:,3:4] # take already the column instead of transposing it
print(test)

class NN:

    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.input_size  = n_inputs
        self.hidden_size = n_hidden
        self.output_size = n_outputs
        self.W1  = np.random.randn(n_hidden, n_inputs)
        self.W2  = np.random.randn(n_outputs, n_hidden)
        self.B1    = np.random.randn(n_hidden, 1)/10
        self.B2     = np.random.randn(n_outputs, 1)/10
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

    def __back_propagation(self, X, Y, learning_rate):
        #(dataset, output, 1)
        m = X.shape[1]
        # Backward propagation: calculate dW1, db1, dW2, db2.
        d_Z2 = self.A2 - Y
        d_Z1 = np.multiply(np.dot(self.W2.T, d_Z2), 1 - np.power(self.A1, 2))

        d_B2 = (1 / m) * np.sum(d_Z2, axis=1, keepdims=True)
        d_B1 = (1 / m) * np.sum(d_Z1, axis=1, keepdims=True)

        d_W2 = (1 / m) * np.dot(d_Z2, self.A1.T)
        d_W1 = (1 / m) * np.dot(d_Z1, X.T)

        # Updating the parameters according to algorithm
        self.W1 = self.W1 - learning_rate * d_W1
        self.B1 = self.B1 - learning_rate * d_B1
        self.W2 = self.W2 - learning_rate * d_W2
        self.B2 = self.B2 - learning_rate * d_B2


    def train(self, dataset, label, epochs):
        costs = []
        for epoch in range(epochs):
            output = self.__forward_propagation(dataset)
            cost = self.__cost_function(label)
            costs.append([epoch,cost])
            asd = self.__back_propagation(dataset, label, 0.1)
        # # costs.np.array(costs)
        # plt.plot(costs[:, 1], costs[:, 0])
        self.test_prediction(test)
        plt.scatter(*zip(*costs))
        plt.ylabel("cost")
        plt.xlabel("epoch")
        plt.show()

    def test_prediction(self,x):
        print("the input for the test is:",x)
        Z1 = np.dot(self.W1, x) + self.B1
        self.A1 = self.__sigmoid(Z1)
        # sigm( (W2 * A1) + B2 ) = A2
        Z2 = np.dot(self.W2, self.A1) + self.B2
        output = self.__sigmoid(Z2)
        print("the output is",np.round(output,3))
        return self.A2


def main():
    neural_network = NN(INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER)

    mean_time = 0
    start_time = datetime.now()
    print("start NN training with",EPOCHS, "epochs")
    neural_network.train(DATASET, LABEL, EPOCHS)
    end_time = datetime.now() - start_time
    mean_time += end_time.total_seconds()
    print("training completed in", round(mean_time,4))

    #print("Time spent: " + str(mean_time/10))


if __name__ == '__main__':
    main()