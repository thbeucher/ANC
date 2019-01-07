import gym
import time
import gym_ple
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import tensorflow.contrib.eager as tfe


def generate_fake_data(num_data=50):
  '''
  Vector of five integers, 0 or 1
  if majority of 0 labels must be 0 otherwise 1
  '''
  X = [np.random.randint(0, 2, size=(5,)) for _ in range(num_data)]
  y = [np.eye(2)[1] if np.count_nonzero(x) > 2 else np.eye(2)[0] for x in X]
  return X, y


def to_batch(data, batch_size=10):
  return [data[i:i+batch_size] for i in range(0, len(data), batch_size)]


class MainNN(object):
  def __init__(self):
    self.hidden_dense = tf.layers.Dense(10, activation=tf.nn.relu, name='hidden_dense')
    self.output_dense = tf.layers.Dense(2, name='output_dense')
    self.layers = [self.hidden_dense, self.output_dense]
    self.optimizer = tf.train.AdamOptimizer()
    self.init_NN()

  def init_NN(self):
    x = tf.convert_to_tensor(np.random.rand(10, 5), tf.float32)
    output = self.forward(x)

  def forward(self, x):
    for layer in self.layers:
      output = layer(x)
      x = output
    return output

  def train(self, grads_n_vars):
    vars = [var for layer in self.layers for var in layer.variables]
    grads_n_vars = [(gnv[0], v) for gnv, v in zip(grads_n_vars, vars)]
    self.optimizer.apply_gradients(grads_n_vars)
    print('MainNN | {}\n\n'.format(self.output_dense.variables))

  def get(self):
    return [layer.variables for layer in self.layers]


class SubNN(object):
  def __init__(self, mainNN, num_epochs=5):
    self.master = mainNN
    self.num_epochs = num_epochs
    self.hidden_dense = tf.layers.Dense(10, activation=tf.nn.relu, name='s_hidden_dense')
    self.output_dense = tf.layers.Dense(2, name='s_output_dense')
    self.layers = [self.hidden_dense, self.output_dense]
    self.optimizer = tf.train.AdamOptimizer()
    self.init_NN()

  def init_NN(self):
    x = tf.convert_to_tensor(np.random.rand(10, 5), tf.float32)
    output = self.forward(x)

  def forward(self, x):
    for layer in self.layers:
      output = layer(x)
      x = output
    return output

  def get_loss(self, x, y):
    output = self.forward(x)
    loss = tf.losses.mean_squared_error(y, output)
    return loss

  def set(self):
    new_variables = self.master.get()
    for layer, new_variable in zip(self.layers, new_variables):
      self.update_layer(layer, new_variable)

  def update_layer(self, layer_to_update, new_values):
    old_kernel, old_bias = layer_to_update.variables
    new_kernel, new_bias = new_values
    old_kernel.assign(new_kernel)
    old_bias.assign(new_bias)

  def return_loss(self, X, Y):
    losses = 0.
    for x, y in zip(X, Y):
      x = tf.convert_to_tensor(x, tf.float32)
      losses += self.get_loss(x, y)
    return losses

  def work(self, idx):
    X, y = generate_fake_data()
    X_batch, y_batch = to_batch(X), to_batch(y)
    for epoch in range(self.num_epochs):
      print('worker {} | Before: {}\n\n'.format(idx, self.output_dense.variables))
      self.set()
      print('worker {} | After: {}\n\n'.format(idx, self.output_dense.variables[0]))
      grads_n_vars = self.optimizer.compute_gradients(lambda: self.return_loss(X_batch, y_batch))
      self.master.train(grads_n_vars)

  def work_step(self, X, y):
    self.set()
    grads_n_vars = self.optimizer.compute_gradients(lambda: self.return_loss(X, y))
    self.master.train(grads_n_vars)


class MyTestShare(object):
  def __init__(self):
    self.shared = 0

  def update(self, v):
    self.shared += v

  def get(self):
    return self.shared


def playing_random(rendering, idx, num_step=10):
  # env = gym.make('SpaceInvaders-v0')
  env = gym.make('FlappyBird-v0')
  env.reset()
  for i in range(num_step):
    if rendering:
      env.render()
      time.sleep(0.1)
    action = env.action_space.sample()
    obs, r, done, info = env.step(action)
    print('gamer = {} | r = {} | done = {}'.format(idx, r, done))
    if done:
      env.reset()
  env.close()


def random_change(idx, share_obj, wait):
  for i in range(3):
    to_add = wait + 1
    print('worker {} | base_val = {} | to_add = {}'.format(idx, share_obj.get(), to_add))
    # time.sleep(wait)
    share_obj.update(to_add)
    print('worker {} | {}'.format(idx, share_obj.get()))


def test_simple_class():
  BaseManager.register('MyTestShare', MyTestShare)
  manager = BaseManager()
  manager.start()
  myts = manager.MyTestShare()
  procs = []
  num_process = mp.cpu_count()
  for i in range(2):
    # p = mp.Process(target=playing_random, args=(False, i))
    p = mp.Process(target=random_change, args=(i, myts, 0.2*(i+1)))
    procs.append(p)
    p.start()

  for process in procs:
    process.join()

  # playing_random(True, 123, num_step=50)
  print(myts.get())

  print('END')


def test_NN_monoprocess_monoslave():
  master = MainNN()
  slave = SubNN(master)
  slave.work()


def test_NN_monoprocess_multislave():
  master = MainNN()
  slave1 = SubNN(master)
  slave2 = SubNN(master)
  X1, y1 = generate_fake_data()
  X2, y2 = generate_fake_data()
  X_batch1, y_batch1 = to_batch(X1), to_batch(y1)
  X_batch2, y_batch2 = to_batch(X2), to_batch(y2)
  for epoch in range(5):
    print('before: {}\n\n'.format(master.output_dense.variables))
    slave1.work_step(X_batch1, y_batch1)
    print('after slave 1: {}\n\n'.format(master.output_dense.variables))
    slave2.work_step(X_batch2, y_batch2)
    input('after slave 2: {}\n\n'.format(master.output_dense.variables))


def test_NN_multiprocess():
  BaseManager.register('MainNN', MainNN)
  manager = BaseManager()
  manager.start()
  master = manager.MainNN()
  procs = []
  num_process = mp.cpu_count()
  workers = []
  for i in range(2):
    worker = SubNN(master)
    workers.append(worker)
    p = mp.Process(target=lambda: worker.work(i))
    procs.append(p)
    p.start()

  for process in procs:
    process.join()


if __name__ == '__main__':
  tfe.enable_eager_execution()

  # test_simple_class()

  # test_NN_monoprocess_monoslave()
  # test_NN_monoprocess_multislave()
  test_NN_multiprocess()
