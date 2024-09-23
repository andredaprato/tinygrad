import unittest
from tinygrad import Tensor
from tinygrad.dtype import dtypes
from test.helpers import get_stats

class TestMemoryCount(unittest.TestCase):
  def test_add(self):
    a = Tensor.empty(1024, 1024, dtype=dtypes.uint8)
    b = Tensor.empty(1024, 1024, dtype=dtypes.uint8)
    _, mem = get_stats(a+b)
    self.assertEqual(mem, 1024*1024*3)  # 2 reads + 1 write

  def test_add_const(self):
    a = Tensor.empty(1024, 1024, dtype=dtypes.uint8)
    _, mem = get_stats(a+3)
    self.assertEqual(mem, 1024*1024*2)  # 1 read + 1 write

  def test_add_slice(self):
    a = Tensor.empty(1024, 1024, dtype=dtypes.uint8)[:512]
    _, mem = get_stats(a+3)
    self.assertEqual(mem, 512*1024*2)  # 1 read + 1 write

  def test_expanded(self):
    a = Tensor.empty(1024, 1, dtype=dtypes.uint8).expand(1024, 1024)
    b = Tensor.empty(1024, 1024, dtype=dtypes.uint8)
    _, mem = get_stats(a+b)
    self.assertEqual(mem, 1024*1024*2 + 1024)  # 1 full read + 1 lil read + 1 write

  def test_both_expanded(self):
    # TODO: this probably should be a full write
    a = Tensor.empty(1024, 1, dtype=dtypes.uint8).expand(1024, 1024)
    b = Tensor.empty(1024, 1, dtype=dtypes.uint8).expand(1024, 1024)
    _, mem = get_stats(a+b)
    self.assertEqual(mem, 1024*1024 + 2*1024)  # 2 lil reads + 1 write

  def test_self_add(self):
    a = Tensor.empty(1024, 1024, dtype=dtypes.uint8)
    _, mem = get_stats(a+a)
    self.assertEqual(mem, 1024*1024*2)  # 1 read + 1 write

  def test_self_add_transposed(self):
    a = Tensor.empty(1024, 1024, dtype=dtypes.uint8)
    _, mem = get_stats(a+a.T)
    self.assertEqual(mem, 1024*1024*2)  # 1 read + 1 write

  def test_self_add_assign(self):
    a = Tensor.empty(1024, 1024, dtype=dtypes.uint8).realize()
    _, mem = get_stats(a.assign(a+a))
    self.assertEqual(mem, 1024*1024*2)  # 1 read + 1 write
