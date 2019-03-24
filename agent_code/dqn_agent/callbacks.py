from settings import s
from .configurations import load_final_configuration


def setup(agent):
    load_final_configuration(agent)


def act(agent):
    agent.next_action = agent.proxy(agent, strings=s.actions)


def reward_update(agent):
    agent.proxy.update(agent)


def end_of_episode(agent):
    agent.proxy.update(agent, last=True)
    agent.proxy.save(agent)

