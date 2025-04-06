import gym

# Tạo môi trường
env = gym.make("LunarLander-v2", render_mode="human")

action_space = env.action_space


if isinstance(action_space, gym.spaces.Discrete):
    num_actions = action_space.n
    # print("Number of actions:", num_actions)
    # print("action",env.action_space.n)


    fixed_action = 3

    observation = env.reset(seed=42)
    for _ in range(100):

        observation, reward, terminated, truncated, info = env.step(fixed_action)
        # print("Reward:", reward)
        print(env.action_space.n)

        if terminated or truncated:
            observation,_ = env.reset()

env.close()
