import pytest

from computational_graph.nodes.multiply import MultiplyNode
from computational_graph.nodes.value import ValueNode


def make_mul(x1_val=None, x2_val=None):
    x1 = ValueNode(x1_val)
    x2 = ValueNode(x2_val)
    out = ValueNode()
    mul = MultiplyNode(x1, x2, out)
    return x1, x2, mul, out


def test_forward_product():
    x1, x2, mul, out = make_mul(3.0, 4.0)
    x1.forward()
    x2.forward()
    assert out.v == 12.0


def test_forward_not_triggered_with_one_input():
    x1, x2, mul, out = make_mul()
    x1.receive_parent_value(3.0)
    x1.forward()
    assert out.v is None
    assert mul.input_ready is False


def test_forward_triggered_after_both_inputs():
    x1, x2, mul, out = make_mul()
    x1.receive_parent_value(3.0)
    x2.receive_parent_value(4.0)
    x1.forward()
    x2.forward()
    assert out.v == 12.0


def test_backward_grad_x1():
    x1, x2, mul, out = make_mul(3.0, 4.0)
    x1.forward()
    x2.forward()
    mul.backward(1.0)
    assert x1.grad_v == 4.0  # dz/dx1 = x2


def test_backward_grad_x2():
    x1, x2, mul, out = make_mul(3.0, 4.0)
    x1.forward()
    x2.forward()
    mul.backward(1.0)
    assert x2.grad_v == 3.0  # dz/dx2 = x1


def test_backward_scaled_grad():
    x1, x2, mul, out = make_mul(2.0, 5.0)
    x1.forward()
    x2.forward()
    mul.backward(3.0)
    assert x1.grad_v == 15.0  # 3 * x2
    assert x2.grad_v == 6.0   # 3 * x1


def test_overflow_receive_raises():
    x1, x2, mul, out = make_mul(1.0, 2.0)
    with pytest.raises(Exception, match="already filled"):
        mul.receive_parent_value(9.0)


def test_reset_values():
    x1, x2, mul, out = make_mul(3.0, 4.0)
    x1.forward()
    x2.forward()
    x1.reset_values()
    assert mul.input_ready is False
    assert mul._received_count == 0
