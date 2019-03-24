from os.path import exists
from torch import load as t_load, save as t_save
import torch
import numpy as np


def environment_extractor(agent):
    from numpy import ones
    # np.array default 17x17 -1 Wall, 0 empty, 1 crate
    arena = agent.game_state['arena']
    # tuple x (int), y (int), name (str), bomb_left (flag), score (int)
    x, y, _, bombs_left, score = agent.game_state['self']
    # list of tuples, tuple contains x(int), y(int), timer(int) [4-0]
    bombs = agent.game_state['bombs']
    bomb_xys = [(x, y) for (x, y, t) in bombs]
    others = [(x, y) for (x, y, n, b, s) in agent.game_state['others']]
    # list of tuples, tuple contains x(int), y(int)
    coins = agent.game_state['coins']

    bomb_map = ones(arena.shape) * 5
    for xb, yb, t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[j, i] = min(bomb_map[j, i], t)

    return arena, (x, y, bombs_left, score), others, coins, bomb_xys, bomb_map


def save_network(net, name):
    t_save(net.state_dict(), f"{name}.pth")


def load_network(net, name):
    path = f"{name}.pth"
    if exists(path):
        net.load_state_dict(t_load(path))
        net.eval()


def minibatch_preparation(net, minibatch, gamma, device):
    environment_batch = torch.cat(tuple(d[0] for d in minibatch)).to(device)
    action_batch = torch.tensor(tuple(d[1] for d in minibatch)).to(device)
    reward_batch = torch.tensor(tuple(d[2] for d in minibatch)).to(device)
    next_environment_batch = torch.cat(tuple(d[3] for d in minibatch)).to(device)
    is_last_batch = torch.tensor(tuple(float(d[4]) for d in minibatch)).to(device)

    batch_size = action_batch.shape[0]
    action_one_hot = torch.zeros((batch_size, 6), device=device).byte()
    action_one_hot[np.arange(batch_size), action_batch] = 1
    q_value = net(environment_batch).masked_select(action_one_hot)

    estimated_reward_batch = net(next_environment_batch).max(1)[0]

    y_batch = reward_batch + gamma * estimated_reward_batch * (1. - is_last_batch)
    y_batch = y_batch.detach()

    return q_value, y_batch
