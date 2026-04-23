from __future__ import annotations

from computational_graph.nodes.meta import MetaNode
from computational_graph.nodes.value import ValueNode


class MultiplyNode(MetaNode):
    """A binary multiplication node: z = x1 * x2.

    Parent roles are fixed at construction (``parents[0]`` is always x1,
    ``parents[1]`` is always x2), making gradient flow deterministic.

    Attributes:
        _received_count (int): Number of parent values received in the current pass.
    """

    def __init__(self, in1: ValueNode, in2: ValueNode, out: ValueNode, name: str = ""):
        """Create a multiplication operator node and wire up connections.

        Args:
            in1 (ValueNode): First multiplicand node.
            in2 (ValueNode): Second multiplicand node.
            out (ValueNode): Output node that receives ``x1 * x2``.
            name (str): Human-readable identifier for this node.
        """
        super().__init__(name)
        # parents[0] is always x1, parents[1] is always x2
        in1.connect_to(self)
        in2.connect_to(self)
        self.connect_to(out)
        self._received_count = 0

    def get_parent_values(self) -> tuple[float, float]:
        """Return the current values of both parent nodes.

        Returns:
            tuple[float, float]: ``(x1, x2)`` values in construction order.
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
        """Compute the product and push it to children once both inputs are ready."""
        if self.input_ready:
            x1_val, x2_val = self.get_parent_values()
            z = x1_val * x2_val
            for node in self.children:
                node.receive_parent_value(z)
                node.forward()

    def backward(self, grad_z: float):
        """Apply the product rule and route gradients to both parent nodes.

        Args:
            grad_z (float): Gradient of the loss with respect to this node's output.
        """
        x1_val, x2_val = self.get_parent_values()
        self.parents[0].backward(grad_z * x2_val)
        self.parents[1].backward(grad_z * x1_val)

    def __repr__(self) -> str:
        return f"MultiplyNode(in1={self.parents[0].name}, in2={self.parents[1].name}, out={self.children[0].name})"
