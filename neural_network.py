import os
import torch
from torch import nn
import numpy as np

#state = np.ndarray
#state.shape = (210, 160)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(4)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear((210//4) * 40, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 6),
        )
        self.device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def predict_action(self, observations: np.ndarray) -> int:
        array_output = self(torch.tensor(observations, 
                                         dtype=torch.float32).to(self.device))
        return torch.argmax(array_output).item()


def main():
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )

    model = NeuralNetwork()
    input = np.random.rand(1, 1, 210, 160)
    test = model.predict_action(input)
    print(test)
    



if __name__ == "__main__":
    main()
