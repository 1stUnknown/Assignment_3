import gymnasium as gym
import ale_py
import threading
from gymnasium.wrappers import GrayscaleObservation
from neural_network import NeuralNetwork
from saver import saving, loading

nn_results: list[tuple] = []


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

    try:
        while True:
            threads = []

            for index, nn in enumerate(list_of_nn):
                threads.append(threading.Thread(target=play_pong,
                                                args=(nn, index,)))

            # Start all threads.
            for t in threads:
                t.start()

            # Wait for all threads to finish.
            for t in threads:
                t.join()

            # order the results
            sorted_results = sorted(nn_results, key=lambda x: x[1])[-top_x:]
            top_results = sorted_results[-top_x:]
            id_of_top_results = [_[0] for _ in top_results]
            # TODO adjust and apply the top weights
            for index in range(len(list_of_nn)):
                if index in id_of_top_results:
                    continue

                list_of_nn[index].set_weights(top_results[0][2], True)

    except KeyboardInterrupt:
        # save everything the top x
        top_results = sorted(nn_results, key=lambda x: x[1])[-top_x:]
        for index, result in enumerate(top_results):
            saving(result[2], f"{index}")


if __name__ == "__main__":
    main()
