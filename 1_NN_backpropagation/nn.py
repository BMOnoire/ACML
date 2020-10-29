import numpy as np
INPUT_LAYER = 8
HIDDEN_LAYER = 3
OUTPUT_LAYER = 8
def main():
    learning_example = np.eye(8, k=0)
    neural_network = NN(INPUT_LAYER,HIDDEN_LAYER,OUTPUT_LAYER)
class NN:
    def __init__(self, INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER):
        self.input_layer_nodes = INPUT_LAYER
        self.hidden_layer_nodes = HIDDEN_LAYER
        self.output_layer_nodes = OUTPUT_LAYER


if __name__ == '__main__':
    main()
