from __future__ import annotations

from .meta import MetaNode
from .value import ValueNode


class AddNode(MetaNode):
    """A variadic addition node: z = sum(x_i)."""

    def __init__(self, in_nodes: list[ValueNode], out_node: ValueNode):
        """Create an addition operator with an arbitrary number of inputs.

        Args:
            in_nodes: List of input addend nodes.
            out_node: Output node receiving the sum.
        """
        super().__init__()
        for node in in_nodes:
            node.connect_to(self)
        self.connect_to(out_node)
        self.inputs: list[float] = []

    def receive_parent_value(self, v: float):
        """Append one addend and mark ready when all addends are present."""
        self.inputs.append(v)
        if len(self.inputs) == len(self.parents):
            self.input_ready = True
        elif len(self.inputs) > len(self.parents):
            raise Exception('All inputs are already set')

    def _reset_local(self):
        self.inputs = []
        self.input_ready = False

    def forward(self):
        """Compute sum of inputs and push result to children."""
        if self.input_ready:
            s = sum(self.inputs)
            for node in self.children:
                node.receive_parent_value(s)
                node.forward()

    def backward(self, grad_z):
        """Distribute same upstream gradient to all addends."""
        grad_x = 1.0 * grad_z
        for node in self.parents:
            node.backward(grad_x)
