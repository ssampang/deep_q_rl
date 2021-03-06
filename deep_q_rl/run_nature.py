#! /usr/bin/env python
"""
Execute a training run of deep-Q-Leaning with parameters that
are consistent with:

Human-level control through deep reinforcement learning.
Nature, 518(7540):529-533, February 2015

"""

import launcher
import sys

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 2000
    EPOCHS = 200
    STEPS_PER_TEST = 400
    BOARD_SIZE = 9

    # ----------------------
    # ALE Parameters
    # ----------------------
    REPEAT_ACTION_PROBABILITY = 0

    # ----------------------
    # Agent/Network parameters:
    # ----------------------
    UPDATE_RULE = 'deepmind_rmsprop'
    BATCH_ACCUMULATOR = 'sum'
    LEARNING_RATE = .00025
    DISCOUNT = .99
    RMS_DECAY = .95 # (Rho)
    RMS_EPSILON = .01
    DEPTH = 3
    MOMENTUM = 0 # Note that the "momentum" value mentioned in the Nature
                 # paper is not used in the same way as a traditional momentum
                 # term.  It is used to track gradient for the purpose of
                 # estimating the standard deviation. This package uses
                 # rho/RMS_DECAY to track both the history of the gradient
                 # and the squared gradient.
    CLIP_DELTA = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = .1
    EPSILON_DECAY = 1000000
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 128
    NETWORK_TYPE = "go_dnn_3"
    FREEZE_INTERVAL = 10000
    REPLAY_START_SIZE = 1000
    DETERMINISTIC = True
    CUDNN_DETERMINISTIC = False

if __name__ == "__main__":
    launcher.launch([], Defaults, __doc__)
    
