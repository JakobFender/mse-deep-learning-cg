from computational_graph.nodes.arithmetic.square import SquareNode
from computational_graph.nodes.value import ValueNode


def make_sq(x_val=None):
    x = ValueNode("x", x_val)
    out = ValueNode("out")
    sq = SquareNode(x, out)
    return x, sq, out


class TestForward:
    def test_squares_value(self):
        x, sq, out = make_sq(3.0)
        x.forward()
        assert out.v == 9.0

    def test_zero(self):
        x, sq, out = make_sq(0.0)
        x.forward()
        assert out.v == 0.0

    def test_negative(self):
        x, sq, out = make_sq(-4.0)
        x.forward()
        assert out.v == 16.0

    def test_not_triggered_without_input(self):
        x, sq, out = make_sq()
        sq.forward()
        assert out.v is None


class TestBackward:
    def test_grad(self):
        x, sq, out = make_sq(3.0)
        x.forward()
        sq.backward(1.0)
        assert x.grad_v == 6.0  # 2 * x

    def test_scaled_grad(self):
        x, sq, out = make_sq(3.0)
        x.forward()
        sq.backward(2.0)
        assert x.grad_v == 12.0  # 2 * x * grad_z = 2*3*2

    def test_zero_input(self):
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
