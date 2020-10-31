import numpy as np

INPUT_LAYER = 8
HIDDEN_LAYER = 3
OUTPUT_LAYER = 8


class NN:

    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.input_size  = n_inputs
        self.hidden_size = n_hidden
        self.output_size = n_outputs
        self.weights_0L  = np.random.randn(n_inputs, n_hidden)
        self.weights_1L  = np.random.randn(n_hidden, n_outputs)
        self.bias_0L     = np.ones((1, n_hidden))
        self.bias_1L     = np.ones((1, n_outputs))


    def __sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


    def __derivate_sigmoid(self, x):
        return (np.exp(-x)) / ((np.exp(-x) + 1.0) ** 2)


    def forward_propagation(self, input):
        if len(input) != self.input_size:
            print("ERROR: input size is not suitable for NN")
            return None
        hidden_layer = self.__sigmoid(input)
        hidden_layer = np.dot(input, self.weights_0L) + self.bias_0L
        hidden_layer = self.__sigmoid(hidden_layer[0])
        output = np.dot(hidden_layer, self.weights_1L) + self.bias_1L
        output = self.__sigmoid(output[0])
        for a in output:
            print(a)
        return output


def main():
    dataset = np.eye(INPUT_LAYER)

    neural_network = NN(INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER)
    for element in dataset:
        qwe = neural_network.forward_propagation(element)
        print(qwe)



if __name__ == '__main__':
    main()
