from torch import nn


class DQN(nn.Module):
    def __init__(self, in_channels=4, out_channels=6):
        super().__init__()

        self.cnn = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)
                                 , nn.ReLU()
                                 , nn.Conv2d(32, 64, kernel_size=3, stride=1)
                                 , nn.ReLU()
                                 , nn.Conv2d(64, 64, kernel_size=3, stride=1)
                                 , nn.ReLU())

        self.classifier = nn.Sequential(nn.Linear(4*4*64, 170)
                                        , nn.ReLU()
                                        , nn.Linear(170, out_channels))

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        return out

    def act(self, x):
        return self(x)

    def predict(self, x):
        return self(x).max(1)[0]

    def update(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class DQN2(nn.Module):
    def __init__(self, in_channels=4, out_channels=6):
        super().__init__()

        self.cnn = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)
                                 , nn.ReLU()
                                 , nn.Conv2d(32, 64, kernel_size=3, stride=1)
                                 , nn.ReLU()
                                 , nn.Conv2d(64, 64, kernel_size=3, stride=1)
                                 , nn.ReLU())

        self.classifier = nn.Sequential(nn.Linear(4*4*64, 256)
                                        , nn.ReLU()
                                        , nn.Linear(256, out_channels))

    def forward(self, x):
        """
        Convolution     5x17x17 -> 32x8x8 -> 64x6x6 -> 64x4x4
        Flatten         64x4x4 -> 1024
        Fully Connected 1024 -> 256 -> 6

        :param x: input tensor of shape Bx5x17x17
        :return: q-values for the 6 actions per batch dimension
        """
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        return out

    def act(self, x):
        return self(x)

    def predict(self, x):
        return self(x).max(1)[0]

    def update(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class DQN3(nn.Module):
    def __init__(self, in_channels=4, out_channels=6):
        super().__init__()

        self.cnn = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)
                                 , nn.BatchNorm2d(32)
                                 , nn.ReLU()
                                 , nn.Conv2d(32, 64, kernel_size=3, stride=1)
                                 , nn.BatchNorm2d(64)
                                 , nn.ReLU()
                                 , nn.Conv2d(64, 64, kernel_size=3, stride=1)
                                 , nn.BatchNorm2d(64)
                                 , nn.ReLU())

        self.classifier = nn.Sequential(nn.Linear(4*4*64, 256)
                                        , nn.ReLU()
                                        , nn.Linear(256, out_channels))

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        return out

    def act(self, x):
        return self(x)

    def predict(self, x):
        return self(x).max(1)[0]

    def update(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class DDQN(nn.Module):
    def __init__(self, in_channels=4, out_channels=6):
        super().__init__()

        self.cnn = nn.Sequential(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2)
                                 , nn.BatchNorm2d(32)
                                 , nn.ReLU()
                                 , nn.Conv2d(32, 64, kernel_size=3, stride=1)
                                 , nn.BatchNorm2d(64)
                                 , nn.ReLU()
                                 , nn.Conv2d(64, 64, kernel_size=3, stride=1)
                                 , nn.BatchNorm2d(64)
                                 , nn.ReLU())

        self.action_classifier = nn.Sequential(nn.Linear(4*4*64, 256)
                                               , nn.ReLU()
                                               , nn.Linear(256, out_channels))

        self.value_classifier = nn.Sequential(nn.Linear(4 * 4 * 64, 256)
                                              , nn.ReLU()
                                              , nn.Linear(256, 1))

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        action = self.action_classifier(x)
        value = self.value_classifier(x)

        return value + (action - action.mean(dim=1, keepdim=True))

    def act(self, x):
        return self(x)

    def predict(self, x):
        return self(x).max(1)[0]

    def update(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class DoubleDQN(nn.Module):
    def __init__(self, net, update_freq=10000, **kwargs):
        super().__init__()
        self.q_net = net(**kwargs)
        self.target_net = net(**kwargs)
        self.update_freq = update_freq
        self.update_calls = 0

    def forward(self, x):
        return self.q_net(x)

    def act(self, x):
        return self(x)

    def predict(self, x):
        _, next_state_actions = self.q_net(x).max(1, keepdim=True)
        out = self.target_net(x).gather(1, next_state_actions)
        return out

    def update(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.__update_target()

    def __update_target(self):
        self.update_calls += 1
        if self.update_calls % self.update_freq is 0:
            print("update_target")
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.update_calls = 0
