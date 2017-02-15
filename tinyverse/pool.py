"""
A thin wrapper for openAI gym environments that maintains a set of parallel games and has a method to generate
interaction sessions given agent one-step applier function.
"""

import numpy as np
from warnings import warn
import gym
from gym.wrappers import Monitor



# A whole lot of space invaders
class EnvPool(object):
    def __init__(self,
                 agent_step,
                 make_env=lambda: gym.make("SpaceInvaders-v0"),
                 memory_init=(),
                 n_games=1):
        """
        A thin wrapper for openAI gym environments that maintains a set of parallel games and has
        a method to generate interaction sessions given agent one-step applier function.


        Capable of some auxilary actions like evaluating agent on one game session (See .evaluate()).

        :param agent_step: Function with the same signature as agent.get_react_function().
        :type agent_step: theano.function

        :param make_env: Factory that produces environments OR a name of the gym environment.
            See gym.envs.registry.all()
        :type make_env: function or str

        :param memory_init: an initial value of agent memory (for a single session, not batch)
        :type memory_init: a list/tuple of memory layers

        :param n_games: Number of parallel games. One game by default.
        :type n_games: int

        Also stores:
           - game states (gym environment)
           - prev observations - last agent observations
           - prev memory states - last agent hidden states

        """

        # save properties
        self.make_env = make_env
        self.agent_step = agent_step
        self.memory_init = memory_init

        #create envs
        self.envs = [self.make_env() for _ in range(n_games)]

        # initial observations
        self.prev_observations = [make_env.reset() for make_env in self.envs]

        # initial agent memory (if you use any)
        self.prev_memory_states = [ np.stack([m]*10) for m in memory_init ]

        # whether particular session has just been terminated and needs restarting.
        self.just_ended = [False] * len(self.envs)

    def interact(self, n_steps=100, verbose=False, add_last_observation=True):
        """Generate interaction sessions with ataries (openAI gym atari environments)
        Sessions will have length n_steps. Each time one of games is finished, it is immediately getting reset
        and this time is recorded in is_alive_log (See returned values).

        :param n_steps: Length of an interaction.
        :param verbose: If True, prints small debug message whenever a game gets reloaded after end.
        :param add_last_observation: If True, appends the final state with
                state=final_state,
                action=-1,
                reward=0,
                new_memory_states=prev_memory_states, effectively making n_steps-1 records.

        :returns: observation_log, action_log, reward_log, [memory_logs], is_alive_log, info_log
        :rtype: a bunch of tensors [batch, tick, size...],
                the only exception is info_log, which is a list of infos for [time][batch], None padded tick
        """

        def env_step(i, action):
            """Environment reaction.
            :returns: observation, reward, is_alive, info
            """

            if not self.just_ended[i]:
                new_observation, cur_reward, is_done, info = self.envs[i].step(action)
                if is_done:
                    # Game ends now, will finalize on next tick.
                    self.just_ended[i] = True

                # note: is_alive=True in any case because environment is still alive (last tick alive) in our notation.
                return new_observation, cur_reward, True, info
            else:
                # Reset environment, get new observation to be used on next tick.
                new_observation = self.envs[i].reset()
                # Reset memory for new episode.
                for m_i in range(len(new_memory_states)):
                    new_memory_states[m_i][i] = 0

                if verbose:
                    print("env %i reloaded" % i)

                self.just_ended[i] = False

                return new_observation, 0, False, {'end': True}

        history_log = []

        for i in range(n_steps - int(add_last_observation)):
            res = self.agent_step(self.prev_observations, *self.prev_memory_states)
            actions, new_memory_states = res[0], res[1:]

            new_observations, cur_rewards, is_alive, infos = zip(*map(env_step, range(len(self.envs)), actions))

            # Append data tuple for this tick.
            history_log.append((self.prev_observations, actions, cur_rewards, new_memory_states, is_alive, infos))

            self.prev_observations = new_observations
            self.prev_memory_states = new_memory_states

        if add_last_observation:
            fake_actions = np.array([env.action_space.sample() for env in self.envs])
            fake_rewards = np.zeros(shape=len(self.envs))
            fake_is_alive = np.ones(shape=len(self.envs))
            history_log.append((self.prev_observations, fake_actions, fake_rewards, self.prev_memory_states,
                                fake_is_alive, [None] * len(self.envs)))

        # cast to numpy arrays, dimensions: [batch,time,*whatever]
        observation_log, action_log, reward_log, memories_log, is_alive_log, info_log = zip(*history_log)

        # cast to [batch_i,time_i,dimension]
        observation_log,action_log,reward_log,is_alive_log = [np.array(tensor).swapaxes(0, 1)
                              for tensor in observation_log,action_log,reward_log,is_alive_log]

        # cast to [batch, time, units] for each memory tensor
        memories_log = list(map(lambda mem: np.array(mem).swapaxes(0, 1), zip(*memories_log)))

        return observation_log, action_log, reward_log, memories_log, is_alive_log, info_log


    def evaluate(self, n_games=1, save_path="./records", use_monitor=True, record_video=True, verbose=True,
                 t_max=10000):
        """Plays an entire game start to end, records the logs(and possibly mp4 video), returns reward.

        :param save_path: where to save the report
        :param record_video: if True, records mp4 video
        :return: total reward (scalar)
        """
        env = self.make_env()

        if not use_monitor and record_video:
            raise warn("Cannot video without gym monitor. If you still want video, set use_monitor to True")

        if record_video :
            env = Monitor(env,save_path,force=True)
        elif use_monitor:
            env = Monitor(env, save_path, video_callable=lambda i: False, force=True)

        game_rewards = []
        for _ in range(n_games):
            # initial observation
            observation = env.reset()
            # initial memory, single-sample batch
            prev_memories = [[m] for m in self.memory_init]

            t = 0
            total_reward = 0
            while True:

                res = self.agent_step(observation[None, ...], *prev_memories)
                action, new_memories = res[0], res[1:]

                observation, reward, done, info = env.step(action[0])

                total_reward += reward
                prev_memories = new_memories

                if done or t >= t_max:
                    if verbose:
                        print("Episode finished after {} timesteps with reward={}".format(t + 1, total_reward))
                    break
                t += 1
            game_rewards.append(total_reward)

        env.close()
        del env
        return game_rewards
