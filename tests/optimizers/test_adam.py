from math import sqrt

import pytest

from computational_graph.comp_graph import CompGraph
from computational_graph.nodes.arithmetic.square import SquareNode
from computational_graph.nodes.value import ValueNode
from computational_graph.optimizers.adam import Adam


def make_graph(w_val, lr, beta1=0.9, beta2=0.9, eps=1e-8, trainable=True):
    """z = w^2; w is optionally trainable."""
    w = ValueNode("w", w_val, trainable=trainable)
    out = ValueNode("out")
    SquareNode(w, out)
    g = CompGraph([w], [out])
    opt = Adam(g, lr, beta1=beta1, beta2=beta2, eps=eps)
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
        # w=3, grad=6, beta1=beta2=0.9, passes=1
        # m=0.6, v=3.6; m_hat=6.0, v_hat=36.0; update=0.1*6/(sqrt(36)+eps)≈0.1
        w, out, g, opt = make_graph(3.0, 0.1)
        g.forward([[3.0]])
        g.backward()
        opt.step()
        m = 0.1 * 6.0
        v = 0.1 * 36.0
        m_hat = m / (1 - 0.9**1)
        v_hat = v / (1 - 0.9**1)
        expected = 3.0 - 0.1 * m_hat / (sqrt(v_hat) + 1e-8)
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

    def test_passes_increments_after_each_step(self):
        w, out, g, opt = make_graph(3.0, 0.1)
        assert opt.passes == 1
        g.forward([[3.0]])
        g.backward()
        opt.step()
        assert opt.passes == 2
        do_step(w, g, opt)
        assert opt.passes == 3


class TestMoments:
    def test_first_moment_after_step_one(self):
        w, out, g, opt = make_graph(3.0, 0.1)
        g.forward([[3.0]])
        g.backward()  # grad = 6
        opt.step()
        assert opt.first_moment[w] == pytest.approx((1 - 0.9) * 6.0)  # 0.6

    def test_second_moment_after_step_one(self):
        w, out, g, opt = make_graph(3.0, 0.1)
        g.forward([[3.0]])
        g.backward()  # grad = 6
        opt.step()
        assert opt.second_moment[w] == pytest.approx((1 - 0.9) * 36.0)  # 3.6

    def test_moments_accumulate_on_second_step(self):
        w, out, g, opt = make_graph(3.0, 0.1)
        do_step(w, g, opt)
        m1 = opt.first_moment[w]
        v1 = opt.second_moment[w]
        grad2 = 2 * w.v
        do_step(w, g, opt)
        assert opt.first_moment[w] == pytest.approx(0.9 * m1 + 0.1 * grad2)
        assert opt.second_moment[w] == pytest.approx(0.9 * v1 + 0.1 * grad2**2)


class TestBiasCorrection:
    def test_first_step_correction_factor(self):
        # With passes=1 and beta1=0.9: correction = 1/(1-0.9^1) = 10
        # Without correction, m_hat would be 0.6; with it: 6.0
        w, out, g, opt = make_graph(3.0, 0.1, beta1=0.9, beta2=0.9)
        g.forward([[3.0]])
        g.backward()
        opt.step()
        m_raw = opt.first_moment[w]
        m_hat = m_raw / (1 - 0.9**1)
        assert m_hat == pytest.approx(6.0)  # bias-corrected ≈ original grad

    def test_bias_correction_diminishes_over_many_steps(self):
        # correction factor = 1/(1-beta^t) → 1 as t grows large
        w, out, g, opt = make_graph(3.0, 0.1)
        for _ in range(100):
            do_step(w, g, opt)
        correction = 1 / (1 - 0.9**opt.passes)
        assert correction == pytest.approx(1.0, abs=1e-3)

    def test_step_size_approximately_lr(self):
        # Adam's bias correction makes the effective step size ≈ lr
        # when gradient is constant: m_hat/sqrt(v_hat) = grad/sqrt(grad^2) = 1
        w, out, g, opt = make_graph(3.0, 0.1)
        g.forward([[3.0]])
        g.backward()
        w_before = w.v
        opt.step()
        assert abs(w_before - w.v) == pytest.approx(0.1, rel=1e-5)
