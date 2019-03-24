import torch
from .utils import load_network, save_network


class AgentProxy:
    """ Proxy for the bomberman agent
        This class contains the complete workflow of the agent.
        Act (implemented in __call__):
        raw environment -> [sensor] -> perception -> [policy] -> [net] -> action

        Update (implemented in update, __create_memory)
        memory -> [replay_memory]
        [replay_memory] -> minibatch -> [minibatch_preparation] -> [criterion] -> loss -> #backward -> [optimizer]
        # optional update of memory
        [statistics]

    """

    def __init__(self, net_name, net, criterion, optimizer
                 , environment_extractor, sensor, short_term_memory, replay_memory
                 , exploration_policy, reward_processor, minibatch_preparation
                 , statistics, batch_size=50, device="cpu"):
        """ Setup of the proxy

        :param net_name: name of the network - needed for load and save
        :param net: model which should act as the agents brain
        :param criterion: loss function which is used during training
        :param optimizer: optimizer which is used during training
        :param environment_extractor: extracts the raw environment
        :param sensor: processes raw environment into a perception
        :param short_term_memory: stores the n last raw environments
        :param replay_memory: contains a pool of game steps from which training data is sampled
        :param exploration_policy: exploration policy, which the agent should use
        :param reward_processor: calculates a reward for the given step
        :param minibatch_preparation: processes the minibatch for learning
        :param statistics: stores information about the training
        :param batch_size: size of the minibatch during training
        :param device: device on which the agents logic should run
        """
        self.netName = net_name
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer

        self.environmentExtractor = environment_extractor
        self.sensor = sensor
        self.shortTermMemory = short_term_memory
        self.replayMemory = replay_memory
        self.explorationPolicy = exploration_policy
        self.rewardProcessor = reward_processor
        self.minibatchPreparation = minibatch_preparation

        self.statistics = statistics

        self.batch_size = batch_size
        self.device = torch.device(device) if type(device) is str else device
        load_network(self.net, self.netName)
        net.to(self.device)

    def __call__(self, agent, strings=None):
        """ Implements the decision process of the agent.

        :param agent: 'agent' of the given framework with all metadata
        :param strings: list of name of the action
        :return: index of the action, that the agent should perform or name of this action
        """
        environment = self.environmentExtractor(agent)
        self.shortTermMemory.append(environment)
        envir = self.shortTermMemory()
        perception = self.sensor(envir).to(self.device)
        action = self.explorationPolicy(self.net, perception, device=self.device)

        agent.last_environment = envir
        agent.next_action_index = action.item()

        if strings is not None:
            return strings[action.item()]

        return action.item()

    def update(self, agent, last=False):
        """ Updates the agent with a minibatch of remembered steps after remembering the current step.
        This function appends the current memory of the step to the replay memory and gets then a random
        minibatch sample.
        The minibatch gets processed into the Q-value of this state and its supposed target value. With this
        the loss can be calculated.
        After the backpropagation through the network the current statistics for this step will be appended. If
        this was the last step the statistics will written to a file.

        :param agent: agent: agent: 'agent' of the given framework with all metadata
        :param last: is final step
        :return:
        """
        earnl = self.__create_memory(agent, last)
        self.replayMemory.append(earnl)
        minibatch = self.replayMemory.get_batch(size=self.batch_size)

        q_value, y_batch = self.minibatchPreparation(net=self.net, minibatch=minibatch
                                                     , gamma=agent.upd_gamma, device=self.device)

        loss = self.criterion(input=q_value, target=y_batch)
        self.net.update(loss, self.optimizer)
        from .memory import PriorityReplayMemory
        if isinstance(self.replayMemory, PriorityReplayMemory):
            self.replayMemory.update(q_value - y_batch)
        self.statistics.append_loss(loss.item())

    def __create_memory(self, agent, last):
        """ Creates a memory of the last step
        The memore consist of the current perception of the environment e, the selected action of
        the agent a, the current reward r, the perception of the next environment n and a flag l
        if this memory is a final state

        :param agent: agent: 'agent' of the given framework with all metadata
        :param last: is final state
        :return: 5-tuple containing the memory
        """
        e = self.sensor(agent.last_environment)
        a = agent.next_action_index
        r = self.rewardProcessor(agent)
        n = self.sensor(self.shortTermMemory.combine(self.environmentExtractor(agent)))
        l = last

        return e, a, r, n, l

    def save(self, agent):
        """ Saves the current model, policy, memory and statistics.

        :param agent: 'agent' of the given framework with all metadata
        :return: None
        """
        save_network(self.net, self.netName)
        self.explorationPolicy.save()
        self.explorationPolicy.increase()
        self.replayMemory.save()
        self.statistics.finish_episode(agent)
