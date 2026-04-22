from __future__ import annotations

from .meta import MetaNode
from .value import ValueNode


class SquareNode(MetaNode):
    """A unary square node: z = x^2."""

    def __init__(self, in_node: ValueNode, out_node: ValueNode):
        """Create a square operator node.

        Args:
            in_node: Input node ``x``.
            out_node: Output node receiving ``x^2``.
        """
        super().__init__()
        in_node.connect_to(self)
        self.connect_to(out_node)
        self.x: float = None

    def receive_parent_value(self, v: float):
        """Store input value and mark node as ready."""
        self.x = v
        self.input_ready = True

    def _reset_local(self):
        self.x = None
        self.input_ready = False

    def forward(self):
        """Compute square and propagate result to children."""
        if self.input_ready:
            z = self.x * self.x
            for node in self.children:
                node.receive_parent_value(z)
                node.forward()

    def backward(self, grad_z):
        """Apply derivative of square function: d(x^2)/dx = 2x."""
        grad_x = 2 * self.x * grad_z
        for node in self.parents:
            node.backward(grad_x)


# ######  TO DO  #########
# class MSELossNode(MetaNode):
# Advice: mimic the MultiplyNode and adapt the forward() and
# backward() methods
