import gymnasium as gym
import numpy as np
import math

import ale_py

from gymnasium.wrappers import GrayscaleObservation
from random import choice, randint
from os import listdir
from os.path import isdir
from threading import Thread

from neural_network import NeuralNetwork
from saver import (save_to_json, saving_weights_to_json,
                   loading_from_json, loading_weights_from_json)
from utility import get_top

nn_results: list[tuple[int, float, np.ndarray]] = []


def play_pong(nn: NeuralNetwork,
              id: int,
              num_episodes: int = 5,
              range_of_seed: tuple = (1,100)) -> list[float]:
    global nn_results
    env = gym.make('ALE/Pong-v5')
    env = GrayscaleObservation(env)

    whole_match_reward = 0
    for _ in range(num_episodes):
        done = False
        total_reward: float = 0

        state, info = env.reset(seed=randint(range_of_seed[0], range_of_seed[1]))

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

def loading_weights(amount_of_nns: int, randomness: bool = False) -> list[NeuralNetwork] | None:
    list_of_nn = [NeuralNetwork() for _ in range(amount_of_nns)]

    if isdir("./savedweights"):
        file_names: list[str] = listdir("./savedweights")

        saved_weights: list[list[np.ndarray]] = []
        for name in file_names:
            data = loading_weights_from_json(name)
            saved_weights.append(data)

        for index, weight in enumerate(saved_weights):
            list_of_nn[index].set_weights(weight)

        for index in range(len(saved_weights), amount_of_nns):
            list_of_nn[index].set_weights(choice(saved_weights), randomness)

    return list_of_nn

def main():
    amount_of_nns = 10
    top_x = 3
    num_episodes = 10

    list_of_nn = loading_weights(amount_of_nns, True)
    
    try:
        main_loop(list_of_nn, top_x=top_x, num_episodes=num_episodes)
    except KeyboardInterrupt:
        # TODO save the total rewards between test and validate
        print("[INFO] Exiting after KeyboardInterupt; proceeding to save weights before exiting")
        top_weights, _ = get_top(nn_results, top_x)
        for index, weight in enumerate(top_weights):
            saving_weights_to_json(weight, f"{index}")
        print("[INFO] Exiting after Saving!")


def testing(amount_of_nns: list[NeuralNetwork], top_x: int, num_episodes: int):
    
    list_of_nn = loading_weights(amount_of_nns)

    results = main_loop(list_of_nn, length_of_running_program=100, top_x=top_x,
              num_episodes=num_episodes, modify_weights=False)
    
    save_to_json(results, "savedresults", "testing.json")
    

def validate(amount_of_nns: int, top_x: int, num_episodes: int):
    """
    once done training, validate the neural networks
    """

    list_of_nn = loading_weights(amount_of_nns)

    results = main_loop(list_of_nn, range_of_seed=(101,200),
              length_of_running_program=100, top_x = top_x,
              num_episodes = num_episodes, modify_weights=False)
    
    save_to_json(results, "savedresults", "validation")

def main_loop(list_of_nn: list[NeuralNetwork], 
              range_of_seed: tuple = (1,100), 
              length_of_running_program: int = float("inf"),
              top_x: int = 3, num_episodes: int = 10, 
              modify_weights: bool = True) -> list[float]:

    if top_x >= len(list_of_nn):
        raise ValueError("top x is equal or greater than"
                         + "the amount of neural networks given")


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
        list_of_match_results.extend([item[1] for item in nn_results])
        nn_results.clear()
        i += 1

    return list_of_match_results

def calculate_mean_sd_and_median():
    # Load test results
    testing_results = loading_from_json("savedresults", "testing")
    validation_results = loading_from_json("savedresults", "validation")

    #Sort lists for median
    testing_results.sort()
    validation_results.sort()

    #Calculate mean value
    mean_testing = sum(testing_results)/len(testing_results)
    mean_validation = sum(validation_results)/len(validation_results)

    sd_testing_list = [math.pow(value - mean_testing, 2) for value in testing_results]
    sd_validation_list = [math.pow(value - mean_validation, 2) for value in validation_results]
    sd_testing = math.sqrt(sum(sd_testing_list)/len(testing_results))
    sd_validation = math.sqrt(sum(sd_validation_list)/len(validation_results))

    #Location of median
    testing_median_location = len(testing_results)//2
    validation_median_location = len(testing_results)//2

    print(f"testing mean: {mean_testing:.5f}\nvalidation mean: {mean_validation:.5f}")
    print(f"difference between means: {abs(mean_validation) - abs(mean_testing):.5f}")

    print(f"sd of testing: {sd_testing:.5f}\nsd of validation: {sd_validation:.5f}")

    print(f"testing median: {testing_results[testing_median_location]:.5f}\n"
          + f"validation median: {validation_results[validation_median_location]:.5f}")
    print(f"difference between medians: {abs(validation_results[validation_median_location]) - abs(testing_results[testing_median_location])}")



if __name__ == "__main__":
    # amount_of_nns = 6
    # top_x = 3
    # num_episodes = 10
    # main()
    # testing(amount_of_nns, top_x, num_episodes)
    # validate(amount_of_nns, top_x, num_episodes)
    calculate_mean_sd_and_median()