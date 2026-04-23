import pytest

from computational_graph.core import CompGraph
from computational_graph.nodes.arithmetic.add import AddNode
from computational_graph.nodes.arithmetic.multiply import MultiplyNode
from computational_graph.nodes.arithmetic.square import SquareNode
from computational_graph.nodes.value import ValueNode


def make_multiply_graph():
    """z = x1 * x2"""
    x1, x2, out = ValueNode("x1"), ValueNode("x2"), ValueNode("out")
    MultiplyNode(x1, x2, out)
    return CompGraph([x1, x2], [out]), x1, x2, out


def make_add_graph():
    """z = x1 + x2 + x3"""
    x1, x2, x3, out = ValueNode("x1"), ValueNode("x2"), ValueNode("x3"), ValueNode("out")
    AddNode([x1, x2, x3], out)
    return CompGraph([x1, x2, x3], [out]), x1, x2, x3, out


def make_chained_graph():
    """z = (x1 * x2)^2  — multiply then square"""
    x1, x2 = ValueNode("x1"), ValueNode("x2")
    mid, out = ValueNode("mid"), ValueNode("out")
    MultiplyNode(x1, x2, mid)
    SquareNode(mid, out)
    return CompGraph([x1, x2], [out]), x1, x2, mid, out


class TestForward:
    def test_multiply(self):
        g, x1, x2, out = make_multiply_graph()
        g.forward([3.0, 4.0])
        assert out.v == 12.0

    def test_add(self):
        g, x1, x2, x3, out = make_add_graph()
        g.forward([1.0, 2.0, 3.0])
        assert out.v == 6.0

    def test_wrong_input_count_raises(self):
        g, *_ = make_multiply_graph()
        with pytest.raises(ValueError, match="number of input"):
            g.forward([1.0])

    def test_sets_forwarded_flag(self):
        g, *_ = make_multiply_graph()
        assert g.forwarded is False
        g.forward([1.0, 2.0])
        assert g.forwarded is True


class TestBackward:
    def test_requires_forward_first(self):
        g, *_ = make_multiply_graph()
        with pytest.raises(RuntimeError, match="call forward first"):
            g.backward()

    def test_multiply_grads(self):
        g, x1, x2, out = make_multiply_graph()
        g.forward([3.0, 4.0])
        g.backward()
        assert x1.grad_v == 4.0
        assert x2.grad_v == 3.0

    def test_add_grads(self):
        g, x1, x2, x3, out = make_add_graph()
        g.forward([1.0, 2.0, 3.0])
        g.backward()
        assert x1.grad_v == 1.0
        assert x2.grad_v == 1.0
        assert x3.grad_v == 1.0


class TestChainedGraph:
    def test_chained_forward(self):
        g, x1, x2, mid, out = make_chained_graph()
        g.forward([3.0, 2.0])  # mid = 6, out = 36
        assert mid.v == 6.0
        assert out.v == 36.0

    def test_chained_backward(self):
        g, x1, x2, mid, out = make_chained_graph()
        g.forward([3.0, 2.0])  # mid = 6
        g.backward()
        # dL/dmid = 2*mid = 12; dL/dx1 = 12*x2 = 24; dL/dx2 = 12*x1 = 36
        assert x1.grad_v == pytest.approx(24.0)
        assert x2.grad_v == pytest.approx(36.0)


class TestResetValues:
    def test_reset_clears_state(self):
        g, x1, x2, out = make_multiply_graph()
        g.forward([3.0, 4.0])
        g.backward()
        g.reset_values()
        assert out.v is None
        assert x1.grad_v is None
        assert x2.grad_v is None
        assert g.forwarded is False

    def test_reset_allows_second_forward(self):
        g, x1, x2, out = make_multiply_graph()
        g.forward([3.0, 4.0])
        g.reset_values()
        g.forward([2.0, 5.0])
        assert out.v == 10.0


def test_validate_rejects_non_value_node_input():
    x = ValueNode("x")
    out = ValueNode("out")
    mul = MultiplyNode(x, x, out)
    with pytest.raises(TypeError, match="not a ValueNode"):
        CompGraph([mul], [out])


def test_validate_rejects_non_value_node_output():
    x1, x2, out = ValueNode("x1"), ValueNode("x2"), ValueNode("out")
    mul = MultiplyNode(x1, x2, out)
    with pytest.raises(TypeError, match="not a ValueNode"):
        CompGraph([x1, x2], [mul])
