import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation
import ale_py

import numpy as np

from os import listdir
from os.path import isdir

from threading import Thread

from random import choice

from neural_network import NeuralNetwork
from saver import saving, loading
from utility import get_top

nn_results: list[tuple[int, float, np.ndarray]] = []


def play_pong(nn: NeuralNetwork, id: int) -> None:
    global nn_results
    env = gym.make('ALE/Pong-v5')
    env = GrayscaleObservation(env)
    num_episodes = 5

    whole_match_reward = 0
    for episode in range(num_episodes):
        done = False
        total_reward: float = 0

        state, info = env.reset()

        while not done:

            # this is where you would insert your policy

            # give current states to the NNs
            action = nn.predict_action(state.reshape((1, 1, 210, 160)))

            next_state, reward, done, truncated, info = env.step(action)

            total_reward += float(reward)

            state = next_state

        print(
            f"Episode {episode + 1} finished with total reward: {total_reward}"
            )
        whole_match_reward += total_reward

    # Close the
    env.close()
    nn_results.append((id, whole_match_reward, nn.weights()))


def main():
    amount_of_nns = 3
    top_x = 2

    list_of_nn: list[NeuralNetwork] = [NeuralNetwork() for _ in
                                       range(amount_of_nns)]

    if isdir("./savedweights"):
        file_names: list[str] = listdir("./savedweights")

        saved_weights: list[list[np.ndarray]] = []
        for name in file_names:
            saved_weights.append(loading(name))

        for index, weight in enumerate(saved_weights):
            list_of_nn[index].set_weights(weight)

        for index in range(len(saved_weights), len(nn_results)):
            list_of_nn[index].set_weights(choice(saved_weights), True)

    try:
        while True:
            threads = []

            for index, nn in enumerate(list_of_nn):
                threads.append(Thread(target=play_pong,
                                      args=(nn, index,)))

            # Start all threads.
            for t in threads:
                t.start()

            # Wait for all threads to finish.
            for t in threads:
                t.join()

            # order the results
            top_weights, id_of_top_results = get_top(nn_results, top_x)

            for index in range(len(list_of_nn)):
                if index in id_of_top_results:
                    continue

                list_of_nn[index].set_weights(choice(top_weights), True)

            nn_results.clear()
    except KeyboardInterrupt:
        # save everything the top x
        top_weights, _ = get_top(nn_results, top_x)
        for index, weight in enumerate(top_weights):
            saving(weight, f"{index}")


if __name__ == "__main__":
    main()
