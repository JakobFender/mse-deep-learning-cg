from math import sqrt

import pytest

from computational_graph.comp_graph import CompGraph
from computational_graph.nodes.arithmetic.square import SquareNode
from computational_graph.nodes.value import ValueNode
from computational_graph.optimizers.rms_prop import RMSProp


def make_graph(w_val, lr, decay=0.9, eps=1e-8, trainable=True):
    """z = w^2; w is optionally trainable."""
    w = ValueNode("w", w_val, trainable=trainable)
    out = ValueNode("out")
    SquareNode(w, out)
    g = CompGraph([w], [out])
    opt = RMSProp(g, lr, decay=decay, eps=eps)
    return w, out, g, opt


def do_step(w, g, opt):
    """Run one full forward/backward/step cycle using w's current value."""
    current = w.v
    g.reset_values()
    g.forward([[current]])
    g.backward()
    opt.step()


class TestStep:
    def test_basic_update(self):
        # w=3, grad=6; velocity = 0.1*36 = 3.6; w = 3 - (0.1/sqrt(3.6+eps))*6
        w, out, g, opt = make_graph(3.0, 0.1)
        g.forward([[3.0]])
        g.backward()
        opt.step()
        expected = 3.0 - (0.1 / sqrt(3.6 + 1e-8)) * 6.0
        assert w.v == pytest.approx(expected)

    def test_clears_grad_after_step(self):
        w, out, g, opt = make_graph(3.0, 0.1)
        g.forward([[3.0]])
        g.backward()
        opt.step()
        assert w.grad_v is None

    def test_non_trainable_not_updated(self):
        w, out, g, opt = make_graph(3.0, 0.1, trainable=False)
        g.forward([[3.0]])
        g.backward()
        opt.step()
        assert w.v == pytest.approx(3.0)


class TestVelocity:
    def test_first_step_velocity(self):
        # velocity starts at 0; after one step: velocity = (1-decay) * grad^2
        w, out, g, opt = make_graph(3.0, 0.1, decay=0.9)
        g.forward([[3.0]])
        g.backward()  # grad = 6
        opt.step()
        assert opt.velocity[w] == pytest.approx(0.1 * 36)  # (1-0.9) * 6^2

    def test_velocity_accumulates_across_steps(self):
        w, out, g, opt = make_graph(3.0, 0.1, decay=0.9)
        do_step(w, g, opt)  # velocity = 3.6 after step 1
        v_after_step1 = opt.velocity[w]
        grad2 = (2 * w.v) ** 2
        do_step(w, g, opt)
        expected_v = 0.9 * v_after_step1 + 0.1 * grad2
        assert opt.velocity[w] == pytest.approx(expected_v)

    def test_decay_controls_velocity_weight(self):
        # velocity = (1-decay)*grad^2 on the first step (starting from zero)
        # higher decay → smaller (1-decay) → less weight on current grad → smaller velocity
        w_high, _, g_high, opt_high = make_graph(3.0, 0.1, decay=0.99)
        w_low, _, g_low, opt_low = make_graph(3.0, 0.1, decay=0.5)
        do_step(w_high, g_high, opt_high)  # velocity = 0.01 * 36 = 0.36
        do_step(w_low, g_low, opt_low)  # velocity = 0.50 * 36 = 18.0
        assert opt_high.velocity[w_high] < opt_low.velocity[w_low]


class TestAdaptiveLR:
    def test_large_grad_gets_smaller_effective_lr(self):
        # For z=w^2: larger w → larger grad → larger velocity → smaller step
        w_large, _, g_large, opt_large = make_graph(10.0, 0.1)
        w_small, _, g_small, opt_small = make_graph(1.0, 0.1)
        g_large.forward([[10.0]])
        g_large.backward()
        opt_large.step()
        step_large = 10.0 - w_large.v

        g_small.forward([[1.0]])
        g_small.backward()
        opt_small.step()
        step_small = 1.0 - w_small.v

        # effective lr = lr / sqrt(velocity); larger grad → larger velocity → smaller ratio
        assert (step_large / 20.0) < (step_small / 2.0)  # grad_large=20, grad_small=2

    def test_eps_prevents_division_by_zero(self):
        w, out, g, opt = make_graph(0.0, 0.1, eps=1e-8)
        g.forward([[0.0]])
        g.backward()  # grad = 0; velocity = 0; update = 0 / sqrt(0 + eps) * 0 = 0
        opt.step()
        assert w.v == pytest.approx(0.0)
