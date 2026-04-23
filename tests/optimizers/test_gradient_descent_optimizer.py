import pytest

from computational_graph.comp_graph import CompGraph
from computational_graph.nodes.arithmetic.multiply import MultiplyNode
from computational_graph.nodes.arithmetic.square import SquareNode
from computational_graph.nodes.value import ValueNode
from computational_graph.optimizers.gradient_descent_optimizer import GradientDescentOptimizer


def make_graph(w_val, lr, trainable=True):
    """z = w^2; w is optionally trainable."""
    w = ValueNode("w", w_val, trainable=trainable)
    out = ValueNode("out")
    SquareNode(w, out)
    g = CompGraph([w], [out])
    opt = GradientDescentOptimizer(g, lr)
    return w, out, g, opt


class TestStep:
    def test_updates_trainable_node(self):
        w, out, g, opt = make_graph(3.0, 0.1)
        g.forward([[3.0]])
        g.backward()  # grad_w = 2 * 3 = 6
        opt.step()
        assert w.v == pytest.approx(2.4)  # 3.0 - 6.0 * 0.1

    def test_learning_rate_scales_update(self):
        w, out, g, opt = make_graph(3.0, 0.5)
        g.forward([[3.0]])
        g.backward()  # grad_w = 6
        opt.step()
        assert w.v == pytest.approx(0.0)  # 3.0 - 6.0 * 0.5

    def test_non_trainable_node_not_updated(self):
        w, out, g, opt = make_graph(3.0, 0.1, trainable=False)
        g.forward([[3.0]])
        g.backward()
        opt.step()
        assert w.v == pytest.approx(3.0)

    def test_clears_grad(self):
        w, out, g, opt = make_graph(3.0, 0.1)
        g.forward([[3.0]])
        g.backward()
        opt.step()
        assert w.grad_v is None

    def test_only_trainable_updated_in_multi_input_graph(self):
        w = ValueNode("w", 2.0, trainable=True)
        x = ValueNode("x", 3.0, trainable=False)
        out = ValueNode("out")
        MultiplyNode(w, x, out)
        g = CompGraph([w, x], [out])
        g.forward([[2.0, 3.0]])
        g.backward()  # grad_w = x = 3, grad_x = w = 2
        opt = GradientDescentOptimizer(g, 0.1)
        opt.step()
        assert w.v == pytest.approx(1.7)  # 2.0 - 3.0 * 0.1
        assert x.v == pytest.approx(3.0)  # unchanged
