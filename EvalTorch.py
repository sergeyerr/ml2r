import numpy as np
import sys
import math
import gym
from Base.bp2DPlot import plot_packing_state
from Base.bpReadWrite import ReadWrite
import TestCherry
import torch
from BoxPlacementEnvironment import BoxPlacementEnvironment
from itertools import count
from TestCherry import policy_net, select_action, device, EPS_START, EPS_END, EPS_DECAY, steps_done

policy_net.load_state_dict(torch.load('./policy_net_500.pytorch_model'))
policy_net.eval()

test_instance = 'state_random_big'
state = ReadWrite.read_state('test_instances/{}'.format(test_instance))
env = gym.make('BoxPlacementEnvironment-v0', bpState=state).unwrapped

EPS_START = 0.05
EPS_END = 0.05
EPS_DECAY = 100

def main():
    with torch.no_grad():
        state = torch.from_numpy(env._next_observation().astype(np.float32)).unsqueeze(0)
        for t in count():
            # Select and perform an action
            action = select_action(state)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            if reward == 0: # placement success
                print('count ', t, ' action ', action.item(), ' reward ', reward.item(), ' done ', done, ' ', env.nr_remaining_boxes, ' eps ', EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY))
            if not done:
                next_state = torch.from_numpy(env._next_observation().astype(np.float32)).unsqueeze(0)
            else:
                next_state = None

            # Move to the next state
            state = next_state

            if done:
                plot_packing_state(env.bpState, fname='./vis/{}'.format(test_instance))
                break

if __name__ == '__main__':
    sys.exit(main())