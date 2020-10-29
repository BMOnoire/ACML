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


    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


    def _derivate_sigmoid(self, x):
        pass
        # TODO


    def forward_propagation(self, inputs):
        hidden_layer = np.dot(inputs, self.weights_0L) + self.bias_0L
        hidden_layer = self._sigmoid(hidden_layer[0])
        outputs = np.dot(hidden_layer, self.weights_1L) + self.bias_1L
        outputs = self._sigmoid(outputs[0])
        return outputs


def main():
    dataset = np.eye(INPUT_LAYER)

    neural_network = NN(INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER)



if __name__ == '__main__':
    main()
