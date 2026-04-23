from __future__ import annotations

from ..meta import MetaNode
from ..value import ValueNode


class MSENode(MetaNode):
    def __init__(self, y_pred: ValueNode, y_true: ValueNode, out_node: ValueNode, name: str = ""):
        super().__init__(name)
        y_pred.connect_to(self)
        y_true.connect_to(self)
        self.connect_to(out_node)
        self._received_count = 0

    def get_parent_values(self) -> tuple[float, float]:
        """Return the current values of both parent nodes.

        Returns:
            tuple[float, float]: ``(x1, x2)`` values in construction order.
        """
        return self.parents[0].v, self.parents[1].v

    def receive_parent_value(self, v: float):
        """Store the input value and mark this node as ready.

        Args:
            v (float): Scalar value forwarded by the parent node.
        """
        del v  # value is read from parents[].v in forward/backward
        if self._received_count >= 2:
            raise Exception("This node accepts 2 inputs that are already filled")
        self._received_count += 1
        if self._received_count == 2:
            self.input_ready = True

    def _reset_local(self):
        """Clear the stored input value and readiness flag."""
        self._received_count = 0
        self.input_ready = False

    def forward(self):
        if self.input_ready:
            y_pred, y_true = self.get_parent_values()
            z = (y_true - y_pred) ** 2
            for node in self.children:
                node.receive_parent_value(z)
                node.forward()

    def backward(self, grad_z: float):
        """Apply the derivative of the square function and propagate to the parent.

        Uses d(x^2)/dx = 2x, so the gradient passed upstream is ``2 * x * grad_z``.

        Args:
            grad_z (float): Gradient of the loss with respect to this node's output.
        """
        y_pred, y_true = self.get_parent_values()
        grad_x = 2 * (y_pred - y_true) * grad_z
        self.parents[0].backward(grad_x)

    def __repr__(self) -> str:
        return f"MSENode(y_pred={self.parents[0].name}, y_true={self.parents[1].name}, out={self.children[0].name})"
