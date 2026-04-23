from __future__ import annotations

from ..meta import MetaNode
from ..value import ValueNode


class MSENode(MetaNode):
    """Squared-error loss node: z = (y_true - y_pred)².

    Computes the squared difference between a prediction and a target.
    Gradient is propagated only to ``y_pred``; ``y_true`` is treated as a
    constant target with no learnable parameters.

    Attributes:
        _received_count (int): Number of parent values received in the current pass.
    """

    def __init__(self, y_pred: ValueNode, y_true: ValueNode, out_node: ValueNode, name: str = ""):
        """Create a squared-error node and wire up connections.

        Args:
            y_pred (ValueNode): Predicted value node (``parents[0]``).
            y_true (ValueNode): Target value node (``parents[1]``).
            out_node (ValueNode): Output node that receives ``(y_true - y_pred)²``.
            name (str): Optional human-readable identifier for this node.
        """
        super().__init__(name)
        y_pred.connect_to(self)
        y_true.connect_to(self)
        self.connect_to(out_node)
        self._received_count = 0

    def get_parent_values(self) -> tuple[float, float]:
        """Return the current values of both parent nodes.

        Returns:
            tuple[float, float]: ``(y_pred, y_true)`` in construction order.
        """
        return self.parents[0].v, self.parents[1].v

    def receive_parent_value(self, v: float):
        """Record that one parent has sent its value.

        The value itself is not stored here; it is read directly from
        ``parents[].v`` during ``forward`` and ``backward``.

        Args:
            v (float): Value forwarded by the calling parent (ignored locally).

        Raises:
            ValueError: If both inputs have already been received.
        """
        del v  # value is read from parents[].v in forward/backward
        if self._received_count >= 2:
            raise ValueError("This node accepts 2 inputs that are already filled")
        self._received_count += 1
        if self._received_count == 2:
            self.input_ready = True

    def _reset_local(self):
        """Reset the received-input counter and readiness flag."""
        self._received_count = 0
        self.input_ready = False

    def forward(self):
        """Compute ``(y_true - y_pred)²`` and push it to children once both inputs are ready."""
        if self.input_ready:
            y_pred, y_true = self.get_parent_values()
            z = (y_true - y_pred) ** 2
            for node in self.children:
                node.receive_parent_value(z)
                node.forward()

    def backward(self, grad_z: float, batch_size: int = 1):
        """Propagate the gradient of the loss with respect to ``y_pred``.

        Uses ``d/dy_pred [(y_true - y_pred)²] = -2 * (y_true - y_pred) = 2 * (y_pred - y_true)``,
        so the gradient passed upstream is ``2 * (y_pred - y_true) * grad_z``.

        Args:
            grad_z (float): Gradient of the loss with respect to this node's output.

            batch_size (int): Batch size to scale gradient"""
        y_pred, y_true = self.get_parent_values()
        grad_x = 2 * (y_pred - y_true) * grad_z
        self.parents[0].backward(grad_x, batch_size)

    def __repr__(self) -> str:
        return f"MSENode(y_pred={self.parents[0].name}, y_true={self.parents[1].name}, out={self.children[0].name})"
