import torch
from torch import nn
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(4)
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
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
        self.device = ("cuda"
                       if torch.cuda.is_available()
                       else "mps"
                       if torch.backends.mps.is_available()
                       else "cpu"
                       )
        print(f"NN is using {self.device}")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.flatten(x)
        logits = self.network(x)
        return logits

    def predict_action(self, observations: np.ndarray) -> int:
        array_output = self(torch.tensor(observations,
                                         dtype=torch.float32).to(self.device))
        return torch.argmax(array_output).item()

    def weights(self) -> list[np.ndarray]:
        weights_list: list[np.ndarray] = []
        for layer in self.network:
            if hasattr(layer, "weight"):
                cpu_tensor = layer.weight.detach().cpu()
                weights_list.append(cpu_tensor.numpy())

        return weights_list


def main():

    model = NeuralNetwork()
    input = np.random.rand(1, 1, 210, 160)
    test = model.predict_action(input)
    print(test)


if __name__ == "__main__":
    main()
