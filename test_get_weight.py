from neural_network import NeuralNetwork
import torch


def main():
    model = NeuralNetwork()
    print(model)

    for layer in model.linear_relu_stack:
        if not isinstance(layer, torch.nn.ReLU):
            print(layer.weight)


if __name__ == '__main__':
    main()
