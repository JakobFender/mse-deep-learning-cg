import pytest

from computational_graph.comp_graph import CompGraph
from computational_graph.nodes.arithmetic.multiply import MultiplyNode
from computational_graph.nodes.arithmetic.square import SquareNode
from computational_graph.nodes.value import ValueNode
from computational_graph.optimizers.gradient_descent import GradientDescent


def make_graph(w_val, lr, trainable=True, momentum=0.0):
    """z = w^2; w is optionally trainable."""
    w = ValueNode("w", w_val, trainable=trainable)
    out = ValueNode("out")
    SquareNode(w, out)
    g = CompGraph([w], [out])
    opt = GradientDescent(g, lr, momentum=momentum)
    return w, out, g, opt


def do_step(w, g, opt):
    """Run one full forward/backward/step cycle using w's current value."""
    current = w.v
    g.reset_values()
    g.forward([[current]])
    g.backward()
    opt.step()


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
        opt = GradientDescent(g, 0.1)
        opt.step()
        assert w.v == pytest.approx(1.7)  # 2.0 - 3.0 * 0.1
        assert x.v == pytest.approx(3.0)  # unchanged


class TestMomentum:
    def test_zero_momentum_equals_plain_gd(self):
        w, out, g, opt = make_graph(3.0, 0.1, momentum=0.0)
        g.forward([[3.0]])
        g.backward()  # grad = 2 * 3 = 6
        opt.step()
        assert w.v == pytest.approx(2.4)  # 3.0 - 0.1 * 6

    def test_first_step_same_regardless_of_momentum(self):
        # velocity starts at 0, so the first step is identical to plain GD
        w, out, g, opt = make_graph(3.0, 0.1, momentum=0.9)
        g.forward([[3.0]])
        g.backward()  # grad = 6; velocity = 0.9 * 0 + 6 = 6
        opt.step()
        assert w.v == pytest.approx(2.4)  # 3.0 - 0.1 * 6

    def test_accumulates_velocity_on_second_step(self):
        w, out, g, opt = make_graph(3.0, 0.1, momentum=0.9)
        do_step(w, g, opt)  # step 1: velocity = 6, w = 2.4
        do_step(w, g, opt)  # step 2: grad = 2 * 2.4 = 4.8; velocity = 0.9 * 6 + 4.8 = 10.2 ; w = 2.4 - 0.1 * 10.2
        assert w.v == pytest.approx(1.38)

    def test_momentum_overshoots_plain_gd(self):
        # After two steps, momentum accumulates a larger update than plain GD
        w_mom, _, g_mom, opt_mom = make_graph(3.0, 0.1, momentum=0.9)
        w_gd, _, g_gd, opt_gd = make_graph(3.0, 0.1, momentum=0.0)
        do_step(w_mom, g_mom, opt_mom)
        do_step(w_mom, g_mom, opt_mom)
        do_step(w_gd, g_gd, opt_gd)
        do_step(w_gd, g_gd, opt_gd)
        assert w_mom.v < w_gd.v  # momentum pushed further from the start
