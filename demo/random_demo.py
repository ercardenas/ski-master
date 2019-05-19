import gym
import time
from gym.envs import registration
from gym.utils import play
# env = gym.make('CartPole-v0')

game = "skiing"
name = "Skiing"
nondeterministic = False
obs_type = "image"

registration.register(
    id='{}-v5'.format(name),
    entry_point='gym.envs.atari:AtariEnv',
    kwargs={'game': game, 'obs_type': obs_type, 'repeat_action_probability': 0.25},
    # max_episode_steps=10000,
    nondeterministic=nondeterministic,
)


env = gym.make('Skiing-v5')
# env = gym.make('skiing')


# def callback(obs_t, obs_tp1, action, rew, done, info):
#     return [rew,]
# plotter = play.PlayPlot(callback, 30 * 5, ["reward"])

# play.play(env, callback=plotter.callback)


env.reset()
i = 0
# for _ in range(1000):
print("Action space: {}".format(env.get_action_meanings()))

game_over = False
rewards = []
while not game_over:
    env.render()
    ob, reward, game_over, d = env.step(env.action_space.sample()) # take a random action
    rewards.append(reward)
    time.sleep(0.0001)
    i += 1
    print(
        "ob:, reward: {reward}, game_over: {game_over}, d: {d}".format(
            ob=ob, reward=reward, game_over=game_over, d=d
        ))
    print("iteration {}, step_result: ".format(i))

print("sum_rewards", sum(rewards))

time.sleep(10)
env.close()
