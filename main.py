import gymnasium as gym
import ale_py
from gymnasium.wrappers import GrayscaleObservation
# from homebrew.nn import NeuralNetwork

env = gym.make('ALE/Pong-v5', render_mode='human')
env = GrayscaleObservation(env)

num_episodes = 5

for episode in range(num_episodes):
    done = False
    total_reward: float = 0

    state, info = env.reset()

    while not done:
        env.render()

        # this is where you would insert your policy
        action = env.action_space.sample()

        next_state, reward, done, truncated, info = env.step(action)

        total_reward += float(reward)

        state = next_state

    print(f"Episode {episode + 1} finished with total reward: {total_reward}")

# Close the environment
env.close()
