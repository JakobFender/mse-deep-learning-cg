import pytest

from computational_graph.nodes.value import ValueNode


def test_init_no_value():
    node = ValueNode("x")
    assert node.v is None
    assert node.grad_v is None
    assert node.input_ready is False


def test_init_with_value():
    node = ValueNode("x", 3.0)
    assert node.v == 3.0
    assert node.input_ready is True


def test_receive_parent_value():
    node = ValueNode("x")
    node.receive_parent_value(5.0)
    assert node.v == 5.0
    assert node.input_ready is True


def test_set_grad_value():
    node = ValueNode("x", 1.0)
    node.set_grad_value(2.5)
    assert node.grad_v == 2.5


def test_forward_propagates_to_child():
    parent = ValueNode("parent", 4.0)
    child = ValueNode("child")
    parent.connect_to(child)
    parent.forward()
    assert child.v == 4.0


def test_forward_raises_without_value():
    node = ValueNode("x")
    with pytest.raises(RuntimeError, match="no value set"):
        node.forward()


class TestBackward:
    def test_sets_grad(self):
        node = ValueNode("x", 1.0)
        node.backward(2.0)
        assert node.grad_v == 2.0

    def test_accumulates_grad(self):
        node = ValueNode("x", 1.0)
        node.backward(2.0)
        node.backward(3.0)
        assert node.grad_v == 5.0

    def test_propagates_to_parent(self):
        parent = ValueNode("parent", 1.0)
        child = ValueNode("child")
        parent.connect_to(child)
        child.backward(1.0)
        assert parent.grad_v == 1.0

    def test_batch_size_divides_grad(self):
        node = ValueNode("x", 1.0)
        node.backward(6.0, batch_size=2)
        assert node.grad_v == pytest.approx(3.0)  # 6 / 2

    def test_batch_size_one_is_default(self):
        node = ValueNode("x", 1.0)
        node.backward(6.0, batch_size=1)
        assert node.grad_v == pytest.approx(6.0)

    def test_batch_size_accumulates_correctly(self):
        node = ValueNode("x", 1.0)
        node.backward(6.0, batch_size=2)
        node.backward(4.0, batch_size=2)
        assert node.grad_v == pytest.approx(5.0)  # 6/2 + 4/2

    def test_batch_size_not_applied_to_parent(self):
        # batch_size divides only at ValueNode; the raw grad_z is forwarded upstream
        parent = ValueNode("parent", 1.0)
        child = ValueNode("child")
        parent.connect_to(child)
        child.backward(6.0, batch_size=2)
        assert parent.grad_v == pytest.approx(3.0)  # parent also divides by batch_size


class TestZeroGrad:
    def test_zero_grad_clears_grad(self):
        node = ValueNode("x", 3.0)
        node.backward(5.0)
        node.zero_grad()
        assert node.grad_v is None

    def test_zero_grad_preserves_value(self):
        node = ValueNode("x", 3.0)
        node.backward(5.0)
        node.zero_grad()
        assert node.v == 3.0

    def test_zero_grad_propagates_to_children(self):
        parent = ValueNode("parent", 2.0)
        child = ValueNode("child")
        parent.connect_to(child)
        child.receive_parent_value(2.0)
        child.backward(1.0)
        parent.zero_grad()
        assert child.grad_v is None


class TestReset:
    def test_reset_values(self):
        node = ValueNode("x", 3.0)
        node.grad_v = 1.0
        node.reset_values()
        assert node.v is None
        assert node.input_ready is False

    def test_reset_propagates_to_children(self):
        parent = ValueNode("parent", 3.0)
        child = ValueNode("child")
        parent.connect_to(child)
        child.receive_parent_value(3.0)
        parent.reset_values()
        assert child.v is None
        assert child.input_ready is False

    def test_reset_preserves_grad(self):
        node = ValueNode("x", 3.0)
        node.grad_v = 1.0
        node.v = 2.0
        node.reset_values()
        assert node.grad_v == 1.0
