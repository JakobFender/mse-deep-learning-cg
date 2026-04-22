from __future__ import annotations

from .meta import MetaNode
from .value import ValueNode


class MultiplyNode(MetaNode):
    """A binary multiplication node: z = x1 * x2.

    Semantic role-based: x1 and x2 roles are explicit at
    construction, making gradient flow deterministic and unambiguous.
    """

    def __init__(self, x1: ValueNode, x2: ValueNode, out: ValueNode):
        """Create a multiplication operator node.

        Args:
            x1: First multiplicand node.
            x2: Second multiplicand node.
            out: Output node receiving ``x1 * x2``.
        """
        super().__init__()
        # parents[0] is always x1, parents[1] is always x2
        x1.connect_to(self)
        x2.connect_to(self)
        self.connect_to(out)
        self._received_count = 0

    def get_parent_values(self) -> tuple[float, float]:
        """Return parent values in the fixed ``(x1, x2)`` order."""
        return self.parents[0].v, self.parents[1].v

    def receive_parent_value(self, v: float):
        """Record input arrival from a parent.

        The scalar itself is not stored here because it is read from parent
        nodes during forward/backward; only readiness state is tracked.
        """
        del v  # value is read from parents[].v in forward/backward
        if self._received_count >= 2:
            raise Exception(
                "This node accepts 2 inputs that are already filled"
            )
        self._received_count += 1
        if self._received_count == 2:
            self.input_ready = True

    def _reset_local(self):
        self._received_count = 0
        self.input_ready = False

    def forward(self):
        """Compute product and push to children once both inputs are ready."""
        if self.input_ready:
            x1_val, x2_val = self.get_parent_values()
            z = x1_val * x2_val
            for node in self.children:
                node.receive_parent_value(z)
                node.forward()

    def backward(self, grad_z):
        """Apply product rule and route gradients to both parents."""
        x1_val, x2_val = self.get_parent_values()
        self.parents[0].backward(grad_z * x2_val)
        self.parents[1].backward(grad_z * x1_val)
