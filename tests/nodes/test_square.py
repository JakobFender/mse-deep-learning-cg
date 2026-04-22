from computational_graph.nodes.square import SquareNode
from computational_graph.nodes.value import ValueNode


def make_sq(x_val=None):
    x = ValueNode(x_val)
    out = ValueNode()
    sq = SquareNode(x, out)
    return x, sq, out


def test_forward_squares_value():
    x, sq, out = make_sq(3.0)
    x.forward()
    assert out.v == 9.0


def test_forward_zero():
    x, sq, out = make_sq(0.0)
    x.forward()
    assert out.v == 0.0


def test_forward_negative():
    x, sq, out = make_sq(-4.0)
    x.forward()
    assert out.v == 16.0


def test_forward_not_triggered_without_input():
    x, sq, out = make_sq()
    sq.forward()
    assert out.v is None


def test_backward_grad():
    x, sq, out = make_sq(3.0)
    x.forward()
    sq.backward(1.0)
    assert x.grad_v == 6.0  # 2 * x


def test_backward_scaled_grad():
    x, sq, out = make_sq(3.0)
    x.forward()
    sq.backward(2.0)
    assert x.grad_v == 12.0  # 2 * x * grad_z = 2*3*2


def test_backward_zero_input():
    x, sq, out = make_sq(0.0)
    x.forward()
    sq.backward(1.0)
    assert x.grad_v == 0.0


def test_reset_values():
    x, sq, out = make_sq(3.0)
    x.forward()
    x.reset_values()
    assert sq.x is None
    assert sq.input_ready is False
