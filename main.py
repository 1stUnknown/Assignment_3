import gymnasium as gym
import ale_py
import threading
from gymnasium.wrappers import GrayscaleObservation
from neural_network import NeuralNetwork

nn_results: list[tuple] = []


def play_pong(nn: NeuralNetwork) -> None:
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

            # give current states to the NNs
            # this is where you would insert your policy

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
    nn_results.append((whole_match_reward, nn.weights()))


def main():
    amount_of_nns = 5

    list_of_nn: list[NeuralNetwork] = [NeuralNetwork() for _ in
                                       range(amount_of_nns)]

    threads = []

    for nn in list_of_nn:
        threads.append(threading.Thread(target=play_pong, args=(nn,)))

    # Start all threads.
    for t in threads:
        t.start()

    # Wait for all threads to finish.
    for t in threads:
        t.join()

    print(nn_results)


if __name__ == "__main__":
    main()
