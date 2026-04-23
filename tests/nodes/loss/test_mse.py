import pytest

from computational_graph.nodes.loss.mse import MSENode
from computational_graph.nodes.value import ValueNode


def make_mse(y_pred=None, y_true=None):
    y_pred = ValueNode("y_pred", y_pred)
    y_true = ValueNode("y_true", y_true)
    out = ValueNode("out")
    mse = MSENode(y_pred, y_true, out)
    return y_pred, y_true, mse, out


class TestForward:
    def test_squared_error(self):
        y_pred, y_true, mse, out = make_mse(3.0, 5.0)
        y_pred.forward()
        y_true.forward()
        assert out.v == 4.0  # (5 - 3)^2

    def test_zero_error(self):
        y_pred, y_true, mse, out = make_mse(2.0, 2.0)
        y_pred.forward()
        y_true.forward()
        assert out.v == 0.0

    def test_negative_pred(self):
        y_pred, y_true, mse, out = make_mse(-1.0, 2.0)
        y_pred.forward()
        y_true.forward()
        assert out.v == 9.0  # (2 - (-1))^2

    def test_not_triggered_with_one_input(self):
        y_pred, y_true, mse, out = make_mse(3.0, 5.0)
        y_pred.forward()
        assert out.v is None
        assert mse.input_ready is False


class TestBackward:
    def test_grad_y_pred(self):
        y_pred, y_true, mse, out = make_mse(3.0, 5.0)
        y_pred.forward()
        y_true.forward()
        mse.backward(1.0)
        assert y_pred.grad_v == -4.0  # 2 * (3 - 5) * 1

    def test_scaled_grad(self):
        y_pred, y_true, mse, out = make_mse(3.0, 5.0)
        y_pred.forward()
        y_true.forward()
        mse.backward(2.0)
        assert y_pred.grad_v == -8.0  # 2 * (3 - 5) * 2

    def test_zero_error_grad(self):
        y_pred, y_true, mse, out = make_mse(3.0, 3.0)
        y_pred.forward()
        y_true.forward()
        mse.backward(1.0)
        assert y_pred.grad_v == 0.0


def test_overflow_receive_raises():
    y_pred, y_true, mse, out = make_mse(1.0, 2.0)
    y_pred.forward()
    y_true.forward()
    with pytest.raises(ValueError, match="already filled"):
        mse.receive_parent_value(9.0)


def test_reset_values():
    y_pred, y_true, mse, out = make_mse(3.0, 5.0)
    y_pred.forward()
    y_true.forward()
    y_pred.reset_values()
    assert mse.input_ready is False
    assert mse._received_count == 0
