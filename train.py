"""Trains agent to play Tic-Tac-Toe.

Uses the defined optimizer procedure to train
the neural network of an agent to play Tic-Tac-Toe.

"""
import random

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.tictactoe import Environment
from src.tictactoe import TicTacToe
from src.model import Model
from src.policy_gradients import PolicyGradients


def train_policy_gradients(env: Environment, model_a: nn.Module, model_b: nn.Module) -> None:
    """Train agents with Policy Gradients."""

    # Trainer
    # Good parameters!
    num_episodes = 500000
    learning_rate = 0.0005
    gamma = 1.0

    agent_a = PolicyGradients(model=model_a, learning_rate=learning_rate, gamma=gamma)
    agent_b = PolicyGradients(model=model_b, learning_rate=learning_rate, gamma=gamma)

    writer = SummaryWriter()

    for episode in range(num_episodes):

        events_a = dict(states=[], actions=[], rewards=[], new_states=[], dones=[])
        events_b = dict(states=[], actions=[], rewards=[], new_states=[], dones=[])

        # Let the agents compete. Rollout one episode.
        if random.random() > 0.5:
            # if episode % 2 == 0:
            events_a, events_b = env.episode(model_a, model_b)
        else:
            events_b, events_a = env.episode(model_b, model_a)

        # Update network.
        loss_a, reward_a = agent_a.step(events_a)
        loss_b, reward_b = agent_b.step(events_b)

        if episode % 500 == 0:
            writer.add_scalar("loss/a", loss_a, episode)
            writer.add_scalar("loss/b", loss_b, episode)
            writer.add_scalar("reward/a", reward_a, episode)
            writer.add_scalar("reward/b", reward_b, episode)

    writer.close()

    print("Model a")
    env.play(model=model_a)
    print("Model b")
    env.play(model=model_b)


if __name__ == "__main__":

    # Playfield
    size = 3

    env = TicTacToe(size=size)

    model_a = Model(size=size)
    model_b = Model(size=size)

    train_policy_gradients(env=env, model_a=model_a, model_b=model_b)
