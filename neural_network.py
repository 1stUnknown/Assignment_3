import torch
import numpy as np
import random
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(4)
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear((210//4) * 40, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )
        self.device = ("cuda"
                       if torch.cuda.is_available()
                       else "mps"
                       if torch.backends.mps.is_available()
                       else "cpu"
                       )
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.flatten(x)
        logits = self.network(x)
        return logits

    def _tensor(self, data: any) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32).to(self.device)

    def predict_action(self, observations: np.ndarray) -> int:
        array_output = self(self._tensor(observations))
        return torch.argmax(array_output).item()

    def weights(self) -> list[np.ndarray]:
        weights_list: list[np.ndarray] = []
        for layer in self.network:
            if hasattr(layer, "weight"):
                cpu_tensor = layer.weight.detach().cpu()
                weights_list.append(cpu_tensor.numpy())

        return weights_list

    def set_weights(self, new_weights: list[np.ndarray],
                    add_randomness: bool = False) -> None:

        for index in range(0, len(self.network), 2):
            weight_index = index // 2
            if add_randomness and random.random() < 0.1:
                random_value = np.random.uniform(
                        -0.1, 0.1,
                        new_weights[weight_index].shape)
            else:
                random_value = np.zeros(
                    new_weights[weight_index].shape)

            # min(1, max(-1, new_weights[index] + random_value))
            self.network[index].weight = nn.Parameter(self._tensor(
                new_weights[weight_index] + random_value))


def main():
    model = NeuralNetwork()
    input = np.random.rand(1, 1, 210, 160)
    test = model.predict_action(input)
    print(test)


if __name__ == "__main__":
    main()
