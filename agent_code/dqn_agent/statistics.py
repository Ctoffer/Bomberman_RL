from os import mkdir
from os.path import exists, join
from numpy import mean
from datetime import datetime


class BasicStatistics:
    """ Appends statistics for each game into a given file

    """
    def __init__(self, fname="stats.txt", mode="Training"):
        """ Initialize the statistics with an path and a tag

        :param fname: path to the statistics file
        :param mode: string for the mode
        """
        self.directory = "statistics"
        self.fname = fname
        self.losses = list()
        self.round = 1
        self.mode = mode
        self.start = datetime.now()

    def append_loss(self, loss):
        """ Saves loss for this step.
        All saved losses will be averaged, when finish_episode gets called

        :param loss: the loss that should be saved
        :return: None
        """
        self.losses.append(loss)

    def finish_episode(self, agent):
        """ Appends one line into the statistic file
        Data ist saved in the format:
        mode | episode | steps | time | loss | score

        :param agent:
        :return:
        """
        if not exists(self.directory):
            mkdir(self.directory)
        path = join(self.directory, self.fname)

        with open(path, 'a') as fp:
            delta = datetime.now() - self.start
            millis = int(delta.total_seconds() * 1000)
            steps, avg, score = len(self.losses), mean(self.losses), agent.game_state['self'][4]
            line = f"{self.mode:>10} | {self.round:>5d} | Steps:{steps:>3d} "
            line += f"| Time:{millis:>5d} | Loss:{avg:>10.5f} | Score:{score:>3d}"
            print(line, file=fp)
            print(line)

        self.round += 1
        self.losses.clear()
        self.start = datetime.now()

