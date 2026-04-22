import pytest

from computational_graph.nodes.value import ValueNode


def test_init_no_value():
    node = ValueNode()
    assert node.v is None
    assert node.grad_v is None
    assert node.input_ready is False


def test_init_with_value():
    node = ValueNode(3.0)
    assert node.v == 3.0
    assert node.input_ready is True


def test_receive_parent_value():
    node = ValueNode()
    node.receive_parent_value(5.0)
    assert node.v == 5.0
    assert node.input_ready is True


def test_set_grad_value():
    node = ValueNode(1.0)
    node.set_grad_value(2.5)
    assert node.grad_v == 2.5


def test_forward_propagates_to_child():
    parent = ValueNode(4.0)
    child = ValueNode()
    parent.connect_to(child)
    parent.forward()
    assert child.v == 4.0


def test_forward_raises_without_value():
    node = ValueNode()
    with pytest.raises(Exception, match="no value set"):
        node.forward()


def test_backward_sets_grad():
    node = ValueNode(1.0)
    node.backward(2.0)
    assert node.grad_v == 2.0


def test_backward_accumulates_grad():
    node = ValueNode(1.0)
    node.backward(2.0)
    node.backward(3.0)
    assert node.grad_v == 5.0


def test_backward_propagates_to_parent():
    parent = ValueNode(1.0)
    child = ValueNode()
    parent.connect_to(child)
    child.backward(1.0)
    assert parent.grad_v == 1.0


def test_reset_values():
    node = ValueNode(3.0)
    node.grad_v = 1.0
    node.reset_values()
    assert node.v is None
    assert node.grad_v is None
    assert node.input_ready is False


def test_reset_propagates_to_children():
    parent = ValueNode(3.0)
    child = ValueNode()
    parent.connect_to(child)
    child.receive_parent_value(3.0)
    parent.reset_values()
    assert child.v is None
    assert child.input_ready is False
