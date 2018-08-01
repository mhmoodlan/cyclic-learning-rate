"""Functional test for learning rate decay."""
import math
#from tensorflow.python.training import learning_rate_decay
import clr as learning_rate_decay
from tensorflow.python.framework import test_util

class CyclicLearningRateTest(test_util.TensorFlowTestCase):
  """Functional test for Cyclic Learning Rate"""
  
  def np_cyclic_learning_rate(self, step, lr, max_lr, step_size, mode):
    cycle = math.floor(1. + step / (2. * step_size))
    x = math.fabs(step / step_size - 2. * cycle + 1.)
    clr = (max_lr - lr) * max(0., 1. - x)
    if mode == 'triangular2':
      clr = clr/(math.pow(2, (cycle - 1)))
    if mode == 'exp_range':
      clr = clr * math.pow(.99994, step)
    return clr + lr
  @test_util.run_in_graph_and_eager_modes()
  def test_triangular(self):
    step = 5
    lr = 0.01
    max_lr = 0.1
    step_size = 20.
    cyclic_lr = learning_rate_decay.cyclic_learning_rate(step,
                                                         lr, max_lr,
                                                         step_size,
                                                         mode='triangular')
    expected = self.np_cyclic_learning_rate(step,
                                            lr, max_lr,
                                            step_size,
                                            mode='triangular')
    self.assertAllClose(self.evaluate(cyclic_lr), expected, 1e-6)
  @test_util.run_in_graph_and_eager_modes()
  def test_triangular2(self):
    step = 5
    lr = 0.01
    max_lr = 0.1
    step_size = 20.
    cyclic_lr = learning_rate_decay.cyclic_learning_rate(step,
                                                         lr, max_lr,
                                                         step_size,
                                                         mode='triangular2')
    expected = self.np_cyclic_learning_rate(step,
                                            lr, max_lr,
                                            step_size,
                                            mode='triangular2')
    self.assertAllClose(self.evaluate(cyclic_lr), expected, 1e-6)
  @test_util.run_in_graph_and_eager_modes()
  def test_exp_range(self):
    step = 5
    lr = 0.01
    max_lr = 0.1
    step_size = 20.
    cyclic_lr = learning_rate_decay.cyclic_learning_rate(step,
                                                         lr, max_lr,
                                                         step_size,
                                                         mode='exp_range')
    expected = self.np_cyclic_learning_rate(step,
                                            lr, max_lr,
                                            step_size,
                                            mode='exp_range')
    self.assertAllClose(self.evaluate(cyclic_lr), expected, 1e-6)
