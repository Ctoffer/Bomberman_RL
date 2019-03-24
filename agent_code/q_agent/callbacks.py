import numpy as np
import os
from settings import e, s, events
from collections import defaultdict


def make_epsilon_greedy_policy(Q, epsilon, noa):
    def policy_fn(game_state):
        A = np.ones(noa, dtype=float) * epsilon / noa
        best_action = np.argmax(Q[game_state])
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def loadQ(path, actions):
    d = {}
    if os.path.exists("Q.npy"):
        d = np.load("Q.npy").item()
    dd = defaultdict(lambda: np.zeros(len(actions)), d)
    return dd

def create_update_function(alpha, discount_factor):
    def update_q(Q, action, state, reward, next_state, actions):
        estimated_action = np.max([Q[next_state][estimated_action] for estimated_action in actions])
        td_target = reward + discount_factor * estimated_action
        td_delta = td_target - Q[state][action]
        Q[state][action] = (1 - alpha) * Q[state][action] + alpha * td_delta

    return update_q


def setup(self):
    self.logger.debug('Successfully entered setup code')
    self.actions = s.actions
    self.event_rewards = [-1, -1, -1, -1, -1, -100, -100, -1, 0, 1, 2, 5, 100, -499, -500, 0, 2]

    self.Q = loadQ("Q.npy", self.actions)

    self.policy = make_epsilon_greedy_policy(self.Q, 0.1, len(self.actions))
    self.episode_data = []
    self.update = create_update_function(alpha=0.5, discount_factor=1.0)

    self.logger.debug('Successfully finished setup code')


def distance(p1, p2):
    return sum([abs(x1 - x2) for x1, x2 in zip(p1, p2)])


def get_game_state(agent):
    arena = agent.game_state['arena']
    x, y, name, bombs_left, score = agent.game_state['self']
    bombs = agent.game_state['bombs']
    bomb_xys = [(x, y) for (x, y, t) in bombs]
    others = [(x, y) for (x, y, n, b, s) in agent.game_state['others']]
    coins = agent.game_state['coins']
    bomb_map = np.zeros(arena.shape)
    for xb, yb, t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    combined_view = np.zeros(arena.shape)
    combined_view += arena
    combined_view += bomb_map - 10

    me = (x, y)
    d_others = tuple(sorted([distance(me, other) for other in others]))
    d_bombs = tuple(sorted([distance(me, bomb) for bomb in bomb_xys]))
    d_coins = tuple(sorted([distance(me, coin) for coin in coins]))
    d_others = -1 if len(d_others) is 0 else d_others[1]
    d_bombs = -1 if len(d_bombs) is 0 else d_bombs[1]
    d_coins = -1 if len(d_coins) is 0 else d_coins[1]

    direct_envir = tuple([combined_view[me[0] + x1][me[1] + x2] for x1, x2 in ((1, 0), (0, 1), (-1, 0), (0, -1))])
    state = (bombs_left, d_others, d_bombs, d_coins, *direct_envir)

    return state


def act(self):
    # Gather information about the game state
    self.logger.debug("Act - Begin")
    merged_state = get_game_state(self)
    self.game_state2 = merged_state
    self.logger.debug(type(self.game_state2))
    action_probs = self.policy(self.game_state2)
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    self.next_action = self.actions[action]
    self.logger.debug("Act - End")
    print("(%s, %s)" % (merged_state, self.next_action), end="")


def update(self):
    # update
    update_q = self.update
    q = self.Q
    action = self.actions.index(self.next_action)
    state = self.game_state2
    reward = sum(self.event_rewards[i] for i in self.events)
    next_state = get_game_state(self)
    actions = [self.actions.index(action) for action in self.actions]
    update_q(q, action, state, reward, next_state, actions)
    print(" -> %s" % [events[e] for e in self.events])


def reward_update(self):
    self.logger.debug("reward_update")

    self.logger.debug(self.events)
    update(self)
    result = dict()
    result["action"] = self.actions.index(self.next_action)
    result["rewards"] = self.event_rewards
    result["state"] = self.game_state2
    result["next_state"] = get_game_state(self)

    self.episode_data.append(result)


def end_of_episode(self):
    self.episode_counter = 0
    self.logger.debug("end_of_episode")
    print([events[e] for e in self.events])
    self.logger.debug(self.events)

    update(self)
    np.save("Q.npy", dict(self.Q))
    print("Saved Q:", len(self.Q))
    self.episode_data.clear()
