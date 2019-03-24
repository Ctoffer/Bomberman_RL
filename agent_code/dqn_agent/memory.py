import torch
import numpy as np
from random import sample


class Singleton:
    def __init__(self, decorated):
        self._decorated = decorated

    def get_instance(self, **kwargs):
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated(**kwargs)
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `get_instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


@Singleton
class CachedReplayMemory:
    """ FiFo replay memory which saves its data to a given directory

    """

    def __init__(self, capacity=10000, directory="cached_rm"):
        self.__pos = 0
        self.__capacity = capacity
        self.__directory = directory
        self.__change_log = list()
        self.__episodes = [None] * self.__capacity
        self.loaded = False

    def append(self, data):
        pos = self.__pos % self.__capacity
        self.__episodes[pos] = data
        if pos in self.__change_log:
            self.save()
        self.__change_log.append(pos)
        self.__pos += 1

    @property
    def episodes(self):
        return self.__episodes

    def get_batch(self, size, device):
        from random import sample

        episodes = [episode for episode in self.__episodes if episode is not None]

        size = min(len(episodes), size)
        minibatch = sample(episodes, size)

        environment_batch = torch.cat(tuple(d[0] for d in minibatch)).float().to(device)
        action_batch = torch.tensor(tuple(d[1] for d in minibatch)).long().to(device)
        reward_batch = torch.tensor(tuple(d[2] for d in minibatch)).float().to(device)
        next_environment_batch = torch.cat(tuple(d[3] for d in minibatch)).float().to(device)
        last_batch = torch.tensor(tuple(int(d[4]) for d in minibatch)).float().to(device)

        return environment_batch, action_batch, reward_batch, next_environment_batch, last_batch

    def save(self):
        from numpy import save as np_save
        from os import mkdir
        from os.path import exists, join

        if not exists(self.__directory):
            mkdir(self.__directory)

        for changed in self.__change_log:
            path = join(self.__directory, f"memory_{changed:05d}")
            np_save(file=path, arr=self.__episodes[changed])

        self.__change_log.clear()

        path = join(self.__directory, f"memory_meta")
        np_save(file=path, arr=[self.__pos, self.__capacity])

    def load(self):
        from numpy import load as np_load
        from os.path import exists, join

        path = join(self.__directory)
        if exists(path):
            path = join(self.__directory, f"memory_meta.npy")
            arr = np_load(file=path)
            self.__pos = arr[0]
            self.__capacity = arr[1]
            self.__episodes = [None] * self.__capacity

            for i in range(len(self.__episodes)):
                path = join(self.__directory, f"memory_{i:05d}.npy")
                if exists(path):
                    self.__episodes[i] = np_load(file=path)
                else:
                    break
        self.loaded = True


class ShortTermMemory:

    def __init__(self, size: int = 4):
        """
        Creates a simple FiFo-Memory for perceptions, that require more
        than one frame
        :param size: size of the internal cache
        """
        self.__size = size
        self.__cache = list()

    def __call__(self):
        """
        Returns the remembered states, if nothing is in memory, the internal
        cache is empty
        :return: list of states or single state if cache size is 1
        """
        if self.__size == 1:
            return self.__cache[0]
        return self.__cache

    def append(self, state):
        """
        Appends a new state into the internal cache, removes the first element
        of the cache to keep demanded size.
        If the cache is empty, it will copy the given state size times
        :param state: new state to remember
        :return: None
        """
        if len(self.__cache) is 0:
            self.__cache = [state] * self.__size
            return

        if len(self.__cache) is self.__size:
            self.__cache.pop(0)
        self.__cache.append(state)

    def combine(self, next_state):
        """
        Returns a version of the internal cache, where the next_state is already
        appended. This does not change the real cache. If internal cache is empty
        a list of size-times next_states will be returned.
        :param next_state: next state, that should be pseudo added
        :return: list of states with next_state or next_state if cache size is 1
        """
        if self.__size == 1:
            return next_state

        if len(self.__cache) is 0:
            return [next_state] * self.__size

        cache = self.__cache.copy()
        cache.pop(0)
        cache.append(next_state)
        return cache


@Singleton
class TemporaryReplayMemory:
    """ simple FiFo runtime memory

    """
    def __init__(self, capacity=10000):
        self.__episodes = []
        self.__capacity = capacity

    def append(self, data):
        if len(self.__episodes) is self.__capacity:
            self.__episodes.pop(0)
        self.__episodes.append(data)

    def get_batch(self, size):
        return sample(self.__episodes, min(len(self.__episodes), size))

    def save(self):
        pass

    def load(self):
        pass


class SumTree:
    """
    Inspired by: https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    """

    def __init__(self, capacity):
        self.__capacity = capacity
        self.__tree = np.zeros(2 * self.__capacity - 1)
        self.__data = np.zeros(self.__capacity, dtype=object)
        self.__data_index = 0

    def summed(self):
        return self.__tree[0]

    def add(self, priority, data):
        index = self.__data_index + self.__capacity - 1

        self.__data[self.__data_index] = data
        self.update(index, priority)

        self.__data_index += 1
        if self.__data_index >= self.__capacity:
            self.__data_index = 0

    def update(self, index, priority):
        change = priority - self.__tree[index]

        self.__tree[index] = priority
        self.__propagate(index, change)

    def __propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.__tree[parent] += change

        if parent != 0:
            self.__propagate(parent, change)

    def get(self, sum):
        index = self.__retrieve(0, sum)
        data_index = index - self.__capacity + 1

        return index, self.__tree[index], self.__data[data_index]

    def __retrieve(self, index, sum):
        left = 2 * index + 1
        right = left + 1

        if left >= len(self.__tree):
            return index

        if sum <= self.__tree[left]:
            return self.__retrieve(left, sum)
        else:
            return self.__retrieve(right, sum - self.__tree[left])


@Singleton
class PriorityReplayMemory:

    def __init__(self, capacity, error_extractor, epsilon=0.01, alpha=0.6):
        """ Implements PER (Priortized Experience Replay)

        :param capacity: maximum capacity of the memory
        :param error_extractor: extractor from data -> error
        :param epsilon: epsilon for priority calculation
        :param alpha: alpha for priority calculation
        """
        self.__eps = epsilon
        self.__alpha = alpha
        self.__capacity = capacity
        self.__tree = SumTree(self.__capacity)
        self.__error_extractor = error_extractor
        self.__cur_size = 0
        self.__last_batch = list()

    def append(self, data):
        """ appends data to the memory

        :param data: data that should be memorized
        :return: None
        """
        if self.__cur_size < self.__capacity:
            self.__cur_size += 1

        self.__tree.add(self.__calc_prio(data), data)

    def __calc_prio(self, data):
        """ Calculate the priority of the data with
        (error + self.eps) ** self.__alpha

        :param data: data for which the priority should be calculated
        :return: priority
        """
        error = self.__error_extractor(data)
        priority = (error + self.__eps) ** self.__alpha
        return priority

    def get_batch(self, size):
        """ Returns a minibatch of size min(cur_size, size)

        :param size: the size the minibatch should maximal have
        :return: minibatch
        """
        sample_indices = np.random.uniform(low=0, high=self.__tree.summed(), size=min(size, self.__cur_size))
        self.__last_batch = [self.__tree.get(_) for _ in sample_indices]
        minibatch = [batch[2] for batch in self.__last_batch]
        return minibatch

    def save(self):
        pass

    def load(self):
        pass

    def update(self, prios):
        """ Updates the priorities of the last minibatch with the given
        priorities.

        :param prios: list of priorities
        :return: None
        """
        for i, info in enumerate(self.__last_batch):
            index, old_prio, data = info
            self.__tree.update(index, abs(prios[i].item()))