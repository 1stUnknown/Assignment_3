import gymnasium as gym
import numpy as np

import ale_py

from gymnasium.wrappers import GrayscaleObservation
from random import choice, randint
from os import listdir
from os.path import isdir
from threading import Thread

from neural_network import NeuralNetwork
from saver import saving, loading
from utility import get_top

nn_results: list[tuple[int, float, np.ndarray]] = []


def play_pong(nn: NeuralNetwork,
              id: int,
              num_episodes: int = 5,
              seed_range: tuple = (0, 100)) -> list[float]:
    global nn_results
    env = gym.make('ALE/Pong-v5')
    env = GrayscaleObservation(env)

    whole_match_reward = 0
    for _ in range(num_episodes):
        done = False
        total_reward: float = 0

        state, info = env.reset(randint(seed_range[0], seed_range[1]))

        while not done:

            # this is where you would insert your policy

            # give current states to the NNs
            action = nn.predict_action(state.reshape((1, 1, 210, 160)))

            next_state, reward, done, truncated, info = env.step(action)

            total_reward += float(reward)

            state = next_state

        whole_match_reward += total_reward
    print(
        f"{id} finished with total reward: {whole_match_reward}"
        )

    # Close the
    nn_results.append((id, whole_match_reward, nn.weights()))
    env.close()


def main():
    global threads
    amount_of_nns = 10
    top_x = 3
    num_episodes = 10

    try:
        main_loop(amount_of_nns=amount_of_nns,
                  top_x=top_x,
                  num_episodes=num_episodes)
    except KeyboardInterrupt:
        print("[INFO] Exiting after KeyboardInterrupt;" +
              " proceeding to save weights before exiting")

        for thread in threads:
            thread.join()

        top_weights, _ = get_top(nn_results, top_x)
        for index, weight in enumerate(top_weights):
            saving(weight, f"{index}")
        print("[INFO] Exiting after Saving!")


def testing() -> float:
    amount_of_nns = 10
    top_x = 3
    num_episodes = 10

    results = main_loop(amount_of_nns=amount_of_nns,
                        length_of_running_program=100,
                        top_x=top_x,
                        num_episodes=num_episodes,
                        modify_weights=False)

    return sum(results)/len(results)


def validate() -> float:
    """
    once done training, validate the neural networks
    """
    amount_of_nns = 10
    top_x = 3
    num_episodes = 10

    results = main_loop(amount_of_nns=amount_of_nns,
                        range_of_seed=(100, 200),
                        length_of_running_program=100,
                        top_x=top_x,
                        num_episodes=num_episodes,
                        modify_weights=False)

    return sum(results)/len(results)


def main_loop(amount_of_nns: int,
              range_of_seed: tuple = (0, 100),
              length_of_running_program: int = float("inf"),
              top_x: int = 3,
              num_episodes: int = 10,
              modify_weights: bool = True) -> list[float]:
    global threads

    if top_x >= amount_of_nns:
        raise ValueError("top x is equal or greater than"
                         + "the amount of neural networks given")

    list_of_nn: list[NeuralNetwork] = [NeuralNetwork() for _ in
                                       range(amount_of_nns)]

    if isdir("./savedweights"):
        file_names: list[str] = listdir("./savedweights")

        saved_weights: list[list[np.ndarray]] = []
        for name in file_names:
            data = loading(name)
            saved_weights.append(data)

        for index, weight in enumerate(saved_weights):
            list_of_nn[index].set_weights(weight)

        for index in range(len(saved_weights), amount_of_nns):
            list_of_nn[index].set_weights(choice(saved_weights), True)

    i = 0
    list_of_match_results = []
    while i < length_of_running_program:
        threads = []

        for index, nn in enumerate(list_of_nn):
            threads.append(Thread(target=play_pong,
                                  args=(nn,
                                        index,
                                        num_episodes,
                                        range_of_seed,)))

        # Start all threads.
        for t in threads:
            t.start()

        # Wait for all threads to finish.
        for t in threads:
            t.join()

        if modify_weights:
            # order the results
            top_weights, ids_of_top_results = get_top(nn_results, top_x)

            for index in range(len(list_of_nn)):
                if index in ids_of_top_results:
                    continue

                list_of_nn[index].set_weights(choice(top_weights), True)
        list_of_match_results.append(item[1] for item in nn_results)

        nn_results.clear()
        i += 1

    return list_of_match_results


if __name__ == "__main__":
    main()
    # testing_results = testing()
    # validation_results = validate()
    # print(validation_results - testing_results)
