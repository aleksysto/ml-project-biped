import gymnasium as gym
from Agent.Agent import Agent

actor = Agent()
actor.critic.load_weights("./critic_1_weights.weights.h5")
env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="rgb_array")
wrapped = gym.wrappers.RecordVideo(env, "./videos2", episode_trigger=lambda episode: (episode+1)>0)
for episode in range(5):
    state, _ = wrapped.reset()
    wrapped.start_video_recorder()
    total_reward = 0
    done = False
    while not done:
        action = actor.critic(state[None, :])[0]
        next, reward, terminated, truncated, info = wrapped.step(action)
        total_reward += reward
        if terminated or truncated:
            done = True
        state = next
    print("Reward: ", total_reward)
    wrapped.close_video_recorder()

wrapped.close()
