import torch
from .utils import load_network, save_network


class DQNAgent:

    def __init__(self, net, sensors, memory, exploration_policy, criterion, optimizer, net_name="dqn_eg_4inp", device="cpu"):
        self.net = net()
        self.net_name = net_name
        self.device = torch.device(device) if type(device) is str else device
        load_network(self.net, self.net_name)

        self.net.to(self.device)
        self.sensors = sensors
        self.replay_memory = memory
        self.exploration_policy = exploration_policy
        self.criterion = criterion
        self.optimizer = optimizer(self.net.parameters())

    def __call__(self, environment):
        perception = self.sensors(environment)
        perception = perception.to(self.device)
        action = self.exploration_policy(self.net, perception, device=self.device)
        return action.item()

    def remember_episode(self, environment, action, reward, next_environment, is_last=False):
        data = environment, action, reward, next_environment, is_last
        self.replay_memory.append(data)

    def update(self, gamma=0.9, batch_size=50):
        minibatch = self.replay_memory.get_batch(size=batch_size, sensors=self.sensors, device=self.device)
        environment_batch = minibatch[0].float()
        action_batch = minibatch[1].float()
        reward_batch = minibatch[2].float()
        next_environment_batch = minibatch[3].float()
        is_last_batch = minibatch[4]

        next_output_batch = self.net(next_environment_batch)

        tmp = []
        for i in range(len(is_last_batch)):
            if is_last_batch[i]:
                tmp.append(reward_batch[i])
            else:
                tmp.append(reward_batch[i] + gamma * torch.max(next_output_batch[i]))

        y_batch = torch.cat(tuple([_.unsqueeze(0).float() for _ in tmp]))

        tmp = self.net(environment_batch)
        tmp2 = torch.zeros(tmp.shape).to(self.device)
        for i in range(tmp2.shape[0]):
            for j in range(tmp2.shape[1]):
                if action_batch[i] == j:
                    tmp2[i, j] = 1

        q_value = torch.sum(tmp * tmp2, dim=1)
        y_batch = y_batch.detach()

        loss = self.criterion(q_value, y_batch)

        self.__update(loss)

    def __update(self, loss):
        print("Loss:", loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save(self):
        save_network(self.net, self.net_name)
        self.exploration_policy.save()
        self.replay_memory.save()


class DQNAgent2:

    def __init__(self, net, sensors, stm, memory, exploration_policy, criterion, optimizer, net_name="dqn_eg_4inp", device="cpu"):
        self.net = net()
        self.net_name = net_name
        self.device = torch.device(device) if type(device) is str else device
        load_network(self.net, self.net_name)

        self.net.to(self.device)
        self.sensors = sensors
        self.stm = stm
        self.replay_memory = memory
        self.exploration_policy = exploration_policy
        self.criterion = criterion
        self.optimizer = optimizer(self.net.parameters())

    def __call__(self, environment):
        envir = self.stm.combine(environment)
        perception = self.sensors(envir)
        perception = perception.to(self.device)
        action = self.exploration_policy(self.net, perception, device=self.device)
        self.exploration_policy.increase()
        return action.item()

    def remember_episode(self, environment, action, reward, next_environment, is_last=False):
        self.stm.append(environment)
        data = self.stm(), action, reward, self.stm.combine(next_environment), is_last
        self.replay_memory.append(data)

    def update(self, gamma=0.9, batch_size=50, print_loss=False):
        minibatch = self.replay_memory.get_batch(size=batch_size, sensors=self.sensors, device=self.device)
        environment_batch = minibatch[0].float()
        action_batch = minibatch[1].float()
        reward_batch = minibatch[2].float()
        next_environment_batch = minibatch[3].float()
        is_last_batch = minibatch[4]

        next_output_batch = self.net(next_environment_batch)

        tmp = []
        for i in range(len(is_last_batch)):
            if is_last_batch[i]:
                tmp.append(reward_batch[i])
            else:
                tmp.append(reward_batch[i] + gamma * torch.max(next_output_batch[i]))

        y_batch = torch.cat(tuple([_.unsqueeze(0).float() for _ in tmp]))

        tmp = self.net(environment_batch)
        tmp2 = torch.zeros(tmp.shape)
        for i in range(tmp2.shape[0]):
            for j in range(tmp2.shape[1]):
                if action_batch[i] == j:
                    tmp2[i, j] = 1
        tmp = tmp.to(self.device)
        tmp2 = tmp2.to(self.device)
        q_value = torch.sum(tmp * tmp2, dim=1)
        y_batch = y_batch.detach()

        loss = self.criterion(q_value, y_batch)

        self.__update(loss, print_loss=print_loss)

    def __update(self, loss, print_loss=False):
        if print_loss:
            print("Loss: {: 10.5f}".format(loss.item()))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def save(self):
        save_network(self.net, self.net_name)
        self.exploration_policy.save()
        self.replay_memory.save()