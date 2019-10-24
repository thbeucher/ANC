import sys
import tensorflow as tf
from hyperparameters import *

sys.path.append(PATH_2_GAME)


class GameEnv(object):
  '''
  This class allows to interact with the game.
  Must provide xxx
  '''
  def __init__(self):
    import wrapped_flappy_bird as game
    flappyBird = game.GameState()
    self.play_function = flappyBird.frame_step


class ActorCriticNetwork(object):
  def __init__(self):
    pass
