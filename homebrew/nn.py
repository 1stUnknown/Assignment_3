from torch import nn
import torch


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2 + 2 + 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def main():
    model = NeuralNetwork().to("cpu")
    print(model)
    X = torch.rand(1, 6, device="cpu")
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")


if __name__ == '__main__':
    main()
