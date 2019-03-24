from numpy import linspace, load as np_load, save as np_save
from os import mkdir
from os.path import join, exists
from random import random, randrange
import torch


class EpsilonGreedyPolicy:

    def __init__(self, name="EGP_1", action_num=6, eps_start=0.9, eps_end=0.05, iters=20000):
        self.num_steps = 0
        self.eps = linspace(eps_start, eps_end, iters)
        self.action_num = action_num
        self.directory = "explorations"
        self.name = name
        self.load()

    def __call__(self, net, environment, device):
        pos = self.num_steps if self.num_steps < len(self.eps) else -1
        max_eps = self.eps[pos]

        if random() > max_eps:
            with torch.no_grad():
                q_values = net(environment)
                q_values = q_values[0]  # extract result - batch of size 1
                q_value, action = torch.max(q_values, 0)
                return action
        else:
            return torch.tensor([[randrange(self.action_num)]], device=device, dtype=torch.long)

    def increase(self):
        self.num_steps += 1

    def save(self):
        if not exists(self.directory):
            mkdir(self.directory)

        arr = [self.num_steps, self.action_num]

        np_save(file=join(self.directory, self.name), arr=arr)
        np_save(file=join(self.directory, self.name + "_eps"), arr=self.eps)

    def load(self):
        path = join(self.directory, self.name + ".npy")

        if exists(path):
            arr = np_load(file=path)
            self.num_steps = arr[0]
            self.action_num = arr[1]

        path = join(self.directory, self.name + "_eps.npy")
        if exists(path):
            self.eps = np_load(file=path)


class TemporaryEpsilonGreedyPolicy:
    def __init__(self, action_num=6, eps_start=0.9, eps_end=0.05, iters=20000):
        self.num_steps = 0
        self.eps = linspace(eps_start, eps_end, iters)
        self.action_num = action_num

    @property
    def pos(self):
        return self.num_steps if self.num_steps < len(self.eps) - 1 else -1

    def __call__(self, net, environment, device):
        if random() > self.eps[self.pos]:
            with torch.no_grad():
                q_values = net(environment)
                q_values = q_values[0]  # extract result - batch of size 1
                q_value, action = torch.max(q_values, 0)
                return action
        else:
            return torch.tensor(randrange(self.action_num), device=device, dtype=torch.long)

    def increase(self):
        self.num_steps += 1

    def save(self):
        pass

    def load(self):
        pass


class ConstantEpsilonGreedyPolicy:
    def __init__(self, action_num=6, eps=0.001):
        self.num_steps = 0
        self.eps = eps
        self.action_num = action_num

    @property
    def pos(self):
        return -1

    def __call__(self, net, environment, device):
        if random() > self.eps:
            with torch.no_grad():
                q_values = net(environment)
                q_values = q_values[0]  # extract result - batch of size 1
                q_value, action = torch.max(q_values, 0)
                return action
        else:
            return torch.tensor(randrange(self.action_num), device=device, dtype=torch.long)

    def increase(self):
        pass

    def save(self):
        pass

    def load(self):
        pass


class PolicyGroup:
    def __init__(self, *args, skip_lower=True):
        self.__current = 0
        self.__skip_lower = skip_lower
        self.__policies = args

    @property
    def current_policy(self):
        return self.__policies[self.__current]

    def __call__(self, net, environment, device):
        return self.current_policy(net, environment, device)

    def increase(self):
        if self.__policy_at_end() and self.__current == len(self.__policies) - 1:
            return

        if self.__policy_at_end():
            self.__current += 1
            if self.__skip_lower:
                self.current_policy.increase()
        else:
            self.current_policy.increase()

    def __policy_at_end(self):
        return self.__policies[self.__current].pos == -1

    @property
    def pos(self):
        return self.current_policy.pos

    def save(self):
        pass

    def load(self):
        pass


if __name__ == '__main__':
    net = lambda envir : torch.zeros((1, 1))
    device = torch.device("cpu")
    environment = None
    p1 = TemporaryEpsilonGreedyPolicy(eps_start=0.9, eps_end=0.1, iters=10)
    p2 = TemporaryEpsilonGreedyPolicy(eps_start=0.1, eps_end=0.01, iters=10)
    p3 = TemporaryEpsilonGreedyPolicy(eps_start=0.01, eps_end=0.001, iters=10)
    mp = PolicyGroup(p1, p2, p3)

    for i in range(30):
        print(i, mp(net, environment, device), mp.pos, "{:5.4f}".format(mp.current_policy.eps[mp.pos]))
        mp.increase()
