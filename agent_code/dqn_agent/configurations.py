from torch.nn import MSELoss
from torch.optim import Adam
import torch

from settings import s

from .explorations import TemporaryEpsilonGreedyPolicy, PolicyGroup, ConstantEpsilonGreedyPolicy
from .memory import ShortTermMemory, TemporaryReplayMemory, PriorityReplayMemory
from .nets import DQN, DQN2, DQN3, DoubleDQN
from .statistics import BasicStatistics
from .proxies import AgentProxy
from .utils import environment_extractor, minibatch_preparation
from .perceptions import CombinedPerception


class RewardProcessor:
    def __init__(self, acc, rewards):
        self.acc = acc
        self.rewards = rewards
        self.reward = 0

    def __call__(self, a):
        if not self.acc:
            self.reward = 0

        self.reward += sum(self.rewards[i] for i in a.events)
        return self.reward


def load_configuration1(agent):
    # 10000
    net_name = "dqn_1_eg_combined"
    stats_name = "stats_1.txt"
    agent.actions = s.actions
    agent.upd_gamma = 0.9
    event_rewards = [-0.1, -0.1, -0.1, -0.1, -0.1, -1, -1, -0.1, 0, 0.2, 0.4, 5.0, 5, -1, -1, 0, 0.2]
    event_rewards = [float(x) for x in event_rewards]

    extractor = environment_extractor
    sensor = CombinedPerception(use_wall=True)
    batch_prep = minibatch_preparation

    net = DQN(in_channels=5, out_channels=6)
    criterion = MSELoss()
    optimizer = Adam(net.parameters(), lr=1e-6)

    egp1 = TemporaryEpsilonGreedyPolicy(eps_start=0.9, eps_end=0.1, iters=4000)
    egp2 = TemporaryEpsilonGreedyPolicy(eps_start=0.1, eps_end=0.01, iters=4000)
    egp3 = TemporaryEpsilonGreedyPolicy(eps_start=0.01, eps_end=0.001, iters=2000)
    egp4 = ConstantEpsilonGreedyPolicy(eps=0.001)
    exploration_policy = PolicyGroup(egp1, egp2, egp3, egp4)
    exploration_policy = ConstantEpsilonGreedyPolicy(eps=0.001)

    short_term_memory = ShortTermMemory(size=5)
    replay_memory = TemporaryReplayMemory.get_instance(capacity=10000)

    reward_processor = RewardProcessor(False, event_rewards)

    statistics = BasicStatistics(fname=stats_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    agent.proxy = AgentProxy(net_name
                             , net
                             , criterion
                             , optimizer
                             , extractor
                             , sensor
                             , short_term_memory
                             , replay_memory
                             , exploration_policy
                             , reward_processor
                             , batch_prep
                             , statistics
                             , batch_size=64
                             , device=device)


def load_configuration2(agent):
    # 30000
    net_name = "dqn_2_eg_combined"
    stats_name = "stats_2.txt"
    agent.actions = s.actions
    agent.upd_gamma = 0.9
    event_rewards = [-0.1, -0.1, -0.1, -0.1, -0.1, -1, -1, -0.1, 0, 0.2, 0.4, 5.0, 5, -1, -1, 0, 0.2]
    event_rewards = [float(x) for x in event_rewards]

    extractor = environment_extractor
    sensor = CombinedPerception(use_wall=True)
    batch_prep = minibatch_preparation

    net = DQN2(in_channels=5, out_channels=6)
    criterion = MSELoss()
    optimizer = Adam(net.parameters(), lr=1e-5)

    # egp1 = TemporaryEpsilonGreedyPolicy(eps_start=0.9, eps_end=0.1, iters=4000)
    # egp2 = TemporaryEpsilonGreedyPolicy(eps_start=0.1, eps_end=0.01, iters=4000)
    # egp3 = TemporaryEpsilonGreedyPolicy(eps_start=0.01, eps_end=0.001, iters=2000)
    # egp4 = TemporaryEpsilonGreedyPolicy(eps_start=0.9, eps_end=0.1, iters=4000)
    # egp5 = TemporaryEpsilonGreedyPolicy(eps_start=0.1, eps_end=0.01, iters=4000)
    # egp6 = TemporaryEpsilonGreedyPolicy(eps_start=0.01, eps_end=0.001, iters=2000)
    # egp7 = TemporaryEpsilonGreedyPolicy(eps_start=0.001, eps_end=0.001, iters=10000)
    egp8 = ConstantEpsilonGreedyPolicy(eps=0.000)
    exploration_policy = PolicyGroup(egp8)

    short_term_memory = ShortTermMemory(size=5)
    replay_memory = TemporaryReplayMemory.get_instance(capacity=10000)

    reward_processor = RewardProcessor(False, event_rewards)

    statistics = BasicStatistics(fname=stats_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent.proxy = AgentProxy(net_name
                             , net
                             , criterion
                             , optimizer
                             , extractor
                             , sensor
                             , short_term_memory
                             , replay_memory
                             , exploration_policy
                             , reward_processor
                             , batch_prep
                             , statistics
                             , batch_size=64
                             , device=device)


def load_configuration3(agent):
    # 10000
    net_name = "dqn_3_eg_combined"
    stats_name = "stats_3.txt"
    agent.actions = s.actions
    agent.upd_gamma = 0.9
    event_rewards = [-0.1, -0.1, -0.1, -0.1, -0.1, -1, -1, -0.1, 0, 0.2, 0.4, 5.0, 5, -1, -1, 0, 0.2]
    event_rewards = [float(x) for x in event_rewards]

    extractor = environment_extractor
    sensor = CombinedPerception(use_wall=True)
    batch_prep = minibatch_preparation

    net = DoubleDQN(DQN2, in_channels=5, out_channels=6)
    criterion = MSELoss()
    optimizer = Adam(net.parameters(), lr=1e-6)

    egp1 = TemporaryEpsilonGreedyPolicy(eps_start=0.9, eps_end=0.1, iters=4000)
    egp2 = TemporaryEpsilonGreedyPolicy(eps_start=0.1, eps_end=0.01, iters=4000)
    egp3 = TemporaryEpsilonGreedyPolicy(eps_start=0.01, eps_end=0.001, iters=2000)
    egp4 = ConstantEpsilonGreedyPolicy(eps=0.001)
    exploration_policy = PolicyGroup(egp1, egp2, egp3, egp4)

    short_term_memory = ShortTermMemory(size=5)
    replay_memory = TemporaryReplayMemory.get_instance(capacity=10000)

    reward_processor = RewardProcessor(False, event_rewards)

    statistics = BasicStatistics(fname=stats_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent.proxy = AgentProxy(net_name
                             , net
                             , criterion
                             , optimizer
                             , extractor
                             , sensor
                             , short_term_memory
                             , replay_memory
                             , exploration_policy
                             , reward_processor
                             , batch_prep
                             , statistics
                             , batch_size=64
                             , device=device)


def load_configuration4(agent):
    #
    net_name = "dqn_4_eg_combined"
    stats_name = "stats_4.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent.actions = s.actions
    agent.upd_gamma = 0.9
    event_rewards = [-0.1, -0.1, -0.1, -0.1, -0.1, -1, -1, -0.1, 0, 0.2, 0.4, 5.0, 5, -1, -1, 0, 0.2]
    event_rewards = [float(x) for x in event_rewards]

    extractor = environment_extractor
    sensor = CombinedPerception(use_wall=True)
    batch_prep = minibatch_preparation

    net = DoubleDQN(DQN2, in_channels=5, out_channels=6)
    criterion = MSELoss()
    optimizer = Adam(net.parameters(), lr=1e-6)

    egp1 = TemporaryEpsilonGreedyPolicy(eps_start=0.9, eps_end=0.1, iters=4000)
    egp2 = TemporaryEpsilonGreedyPolicy(eps_start=0.1, eps_end=0.01, iters=4000)
    egp3 = TemporaryEpsilonGreedyPolicy(eps_start=0.01, eps_end=0.001, iters=2000)
    egp4 = ConstantEpsilonGreedyPolicy(eps=0.001)
    exploration_policy = PolicyGroup(egp1, egp2, egp3, egp4)

    short_term_memory = ShortTermMemory(size=5)

    def error_extractor(memory):
        e, a, r, n, l = memory
        return abs(r)

    replay_memory = PriorityReplayMemory.get_instance(capacity=10000, error_extractor=error_extractor)

    reward_processor = RewardProcessor(False, event_rewards)

    statistics = BasicStatistics(fname=stats_name)

    agent.proxy = AgentProxy(net_name
                             , net
                             , criterion
                             , optimizer
                             , extractor
                             , sensor
                             , short_term_memory
                             , replay_memory
                             , exploration_policy
                             , reward_processor
                             , batch_prep
                             , statistics
                             , batch_size=64
                             , device=device)


def load_configuration5(agent):
    #
    net_name = "dqn_5_eg_combined"
    stats_name = "stats_5.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent.actions = s.actions
    agent.upd_gamma = 0.9
    event_rewards = [-0.2, -0.2, -0.2, -0.2, -0.2, -5, -5, -0.2, 0, 0.2, 0.4, 5.0, 5, -5, -5, 0, 0.0]
    event_rewards = [float(x) for x in event_rewards]

    extractor = environment_extractor
    sensor = CombinedPerception(use_wall=True)
    batch_prep = minibatch_preparation

    net = DoubleDQN(DQN3, in_channels=5, out_channels=6)
    criterion = MSELoss()
    optimizer = Adam(net.parameters(), lr=2.5e-5)

    egp1 = TemporaryEpsilonGreedyPolicy(eps_start=0.9, eps_end=0.1, iters=4000)
    egp2 = TemporaryEpsilonGreedyPolicy(eps_start=0.1, eps_end=0.01, iters=4000)
    egp3 = TemporaryEpsilonGreedyPolicy(eps_start=0.01, eps_end=0.001, iters=2000)
    egp4 = ConstantEpsilonGreedyPolicy(eps=0.001)
    exploration_policy = PolicyGroup(egp1, egp2, egp3, egp4)

    short_term_memory = ShortTermMemory(size=5)

    def error_extractor(memory):
        e, a, r, n, l = memory
        return abs(r)

    replay_memory = PriorityReplayMemory.get_instance(capacity=10000, error_extractor=error_extractor)

    reward_processor = RewardProcessor(False, event_rewards)

    statistics = BasicStatistics(fname=stats_name)

    agent.proxy = AgentProxy(net_name
                             , net
                             , criterion
                             , optimizer
                             , extractor
                             , sensor
                             , short_term_memory
                             , replay_memory
                             , exploration_policy
                             , reward_processor
                             , batch_prep
                             , statistics
                             , batch_size=64
                             , device=device)

def load_configuration6(agent):
    #
    net_name = "dqn_6_eg_combined"
    stats_name = "stats_6.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent.actions = s.actions
    agent.upd_gamma = 0.9
    event_rewards = [-0.1, -0.1, -0.1, -0.1, -0.1, -1, -1, -0.1, 0, 0.2, 0.4, 5.0, 5, -1, -1, 0, 0]
    event_rewards = [float(x) for x in event_rewards]

    extractor = environment_extractor
    sensor = CombinedPerception(use_wall=True)
    batch_prep = minibatch_preparation

    net = DoubleDQN(DQN3, in_channels=5, out_channels=6)
    criterion = MSELoss()
    optimizer = Adam(net.parameters(), lr=1e-6)

    egp1 = TemporaryEpsilonGreedyPolicy(eps_start=0.9, eps_end=0.1, iters=4000)
    egp2 = TemporaryEpsilonGreedyPolicy(eps_start=0.1, eps_end=0.01, iters=4000)
    egp3 = TemporaryEpsilonGreedyPolicy(eps_start=0.01, eps_end=0.001, iters=2000)
    egp4 = ConstantEpsilonGreedyPolicy(eps=0.001)
    exploration_policy = PolicyGroup(egp1, egp2, egp3, egp4)

    short_term_memory = ShortTermMemory(size=5)

    def error_extractor(memory):
        e, a, r, n, l = memory
        return abs(r)

    replay_memory = PriorityReplayMemory.get_instance(capacity=10000, error_extractor=error_extractor)

    reward_processor = RewardProcessor(False, event_rewards)

    statistics = BasicStatistics(fname=stats_name)

    agent.proxy = AgentProxy(net_name
                             , net
                             , criterion
                             , optimizer
                             , extractor
                             , sensor
                             , short_term_memory
                             , replay_memory
                             , exploration_policy
                             , reward_processor
                             , batch_prep
                             , statistics
                             , batch_size=64
                             , device=device)


def load_final_configuration(agent):
    net_name = "./agent_code/dqn_agent/dqn_2_eg_combined_coins_final"
    stats_name = "stats_2.txt"
    agent.actions = s.actions
    agent.upd_gamma = 0.9
    event_rewards = [-0.1, -0.1, -0.1, -0.1, -0.1, -1, -1, -0.1, 0, 0.2, 0.4, 5.0, 5, -1, -1, 0, 0.2]
    event_rewards = [float(x) for x in event_rewards]

    extractor = environment_extractor
    sensor = CombinedPerception(use_wall=True)
    batch_prep = minibatch_preparation

    net = DQN2(in_channels=5, out_channels=6)
    criterion = MSELoss()
    optimizer = Adam(net.parameters(), lr=1e-5)

    exploration_policy = ConstantEpsilonGreedyPolicy(eps=0.000)

    short_term_memory = ShortTermMemory(size=5)
    replay_memory = TemporaryReplayMemory.get_instance(capacity=10000)

    reward_processor = RewardProcessor(False, event_rewards)

    statistics = BasicStatistics(fname=stats_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent.proxy = AgentProxy(net_name
                             , net
                             , criterion
                             , optimizer
                             , extractor
                             , sensor
                             , short_term_memory
                             , replay_memory
                             , exploration_policy
                             , reward_processor
                             , batch_prep
                             , statistics
                             , batch_size=64
                             , device=device)
