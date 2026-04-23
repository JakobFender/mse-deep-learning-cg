from __future__ import annotations

from computational_graph.nodes.meta import MetaNode
from computational_graph.nodes.value import ValueNode


class SquareNode(MetaNode):
    """A unary square node: z = x^2.

    Attributes:
        x (float | None): The input value received from the parent node.
            ``None`` until a value is received.
    """

    def __init__(self, in_node: ValueNode, out_node: ValueNode, name: str = ""):
        """Create a square operator node and wire up connections.

        Args:
            in_node (ValueNode): Input node providing the value to be squared.
            out_node (ValueNode): Output node that receives ``x^2``.
            name (str): Human-readable identifier for this node.
        """
        super().__init__(name)
        in_node.connect_to(self)
        self.connect_to(out_node)
        self.x: float = None

    def receive_parent_value(self, v: float):
        """Store the input value and mark this node as ready.

        Args:
            v (float): Scalar value forwarded by the parent node.
        """
        self.x = v
        self.input_ready = True

    def _reset_local(self):
        """Clear the stored input value and readiness flag."""
        self.x = None
        self.input_ready = False

    def forward(self):
        """Compute the square of the input and propagate the result to children."""
        if self.input_ready:
            z = self.x * self.x
            for node in self.children:
                node.receive_parent_value(z)
                node.forward()

    def backward(self, grad_z: float, batch_size: int = 1):
        """Apply the derivative of the square function and propagate to the parent.

        Uses d(x^2)/dx = 2x, so the gradient passed upstream is ``2 * x * grad_z``.

        Args:
            grad_z (float): Gradient of the loss with respect to this node's output.
            batch_size (int): Batch size to scale gradient
        """
        grad_x = 2 * self.x * grad_z
        for node in self.parents:
            node.backward(grad_x, batch_size)

    def __repr__(self) -> str:
        return f"SquareNode(in={self.parents[0].name}, out={self.children[0].name})"
