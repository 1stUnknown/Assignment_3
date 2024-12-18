import gymnasium as gym

env = gym.make('ALE/Pong-v5', render_mode='human')

state, info = env.reset()

num_episodes = 5

for episode in range(num_episodes):
    done = False
    total_reward = 0

    state, info = env.reset()

    while not done:
        env.render()

         # this is where you would insert your policy
        action = env.action_space.sample()
        action = env.action_space.sample() #RANDOMMMM

        next_state, reward, done, truncated, info = env.step(action)

        total_reward += reward

        state = next_state

    print(f"Episode {episode + 1} finished with total reward: {total_reward}")

# Close the environment
env.close()