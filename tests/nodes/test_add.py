import pytest

from computational_graph.nodes.arithmetic.add import AddNode
from computational_graph.nodes.value import ValueNode


def make_add(*values):
    in_nodes = [ValueNode(f"x{i}", v) for i, v in enumerate(values)]
    out = ValueNode("out")
    add = AddNode(in_nodes, out)
    return in_nodes, add, out


class TestForward:
    def test_two_inputs(self):
        ins, add, out = make_add(2.0, 3.0)
        for node in ins:
            node.forward()
        assert out.v == 5.0

    def test_three_inputs(self):
        ins, add, out = make_add(1.0, 2.0, 3.0)
        for node in ins:
            node.forward()
        assert out.v == 6.0

    def test_not_triggered_until_all_inputs(self):
        ins, add, out = make_add(1.0, 2.0, 3.0)
        ins[0].forward()
        assert out.v is None
        ins[1].forward()
        assert out.v is None
        ins[2].forward()
        assert out.v == 6.0


def test_backward_distributes_grad():
    ins, add, out = make_add(2.0, 3.0)
    for node in ins:
        node.forward()
    add.backward(1.0)
    for node in ins:
        assert node.grad_v == 1.0


def test_backward_scaled_grad():
    ins, add, out = make_add(2.0, 3.0)
    for node in ins:
        node.forward()
    add.backward(4.0)
    for node in ins:
        assert node.grad_v == 4.0


def test_overflow_receive_raises():
    ins, add, out = make_add(1.0, 2.0)
    for node in ins:
        node.forward()
    with pytest.raises(Exception, match="already set"):
        add.receive_parent_value(9.0)


def test_reset_values():
    ins, add, out = make_add(2.0, 3.0)
    for node in ins:
        node.forward()
    ins[0].reset_values()
    assert add.inputs == []
    assert add.input_ready is False
