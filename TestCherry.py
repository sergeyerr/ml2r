#!/usr/bin/env python3

import gym

import torch.nn as nn
import torch.optim as optim

import cherry as ch
import cherry.envs as envs
from cherry.models.tabular import ActionValueFunction
from Base.bpStateGenerators import random_state_generator

from BoxPlacementEnvironment import BoxPlacementEnvironment

class Agent(nn.Module):

    def __init__(self, env):
        super(Agent, self).__init__()
        self.env = env
        self.qf = ActionValueFunction(200,
                                      10001)
        self.e_greedy = ch.nn.EpsilonGreedy(0.1)

    def forward(self, x):
        # x = ch.onehot(x, self.env.state_size)
        q_values = self.qf(x)
        action = self.e_greedy(q_values)
        info = {
            'q_action': q_values[:, action],
        }
        return action, info


def main(env='BoxPlacementEnvironment-v0'):
    gym.register(
        'BoxPlacementEnvironment-v0',
        entry_point = BoxPlacementEnvironment
    )
    test_state = random_state_generator((10, 10),20,1,3,2,8)
    env = gym.make(env, bpState= test_state)
    env = envs.Logger(env, interval=100)
    env = envs.Torch(env)
    env = envs.Runner(env)
    agent = Agent(env)
    discount = 1.00
    optimizer = optim.SGD(agent.parameters(), lr=0.5, momentum=0.0)
    for iteration in range(1000):
        test_state = random_state_generator((10, 10),20,1,3,2,8)
        env.reset(bpState=test_state)
        if iteration % 10 == 0:
            print("BEFORE")
            print(env.bin_occupation)
            print(env.box_counts)
            print("------")
        for t in range(1, 100):
            transition = env.run(agent, steps=1)[0]

            curr_q = transition.q_action
            next_state = transition.next_state
            # next_state = ch.onehot(transition.next_state,
            #                        dim=env.state_size)
            next_q = agent.qf(next_state).max().detach()
            td_error = ch.temporal_difference(discount,
                                            transition.reward,
                                            transition.done,
                                            curr_q,
                                            next_q)

            optimizer.zero_grad()
            loss = td_error.pow(2).mul(0.5)
            loss.backward()
            optimizer.step()
        if iteration % 10 == 0:
            print("AFTER")
            print(env.bin_occupation)
            print(env.box_counts)
            print("------")


if __name__ == '__main__':
    main()