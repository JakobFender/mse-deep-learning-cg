from __future__ import annotations
from typing import Optional

from .meta import MetaNode


class ValueNode(MetaNode):
    """A node that stores scalar values and propagated gradients."""

    def __init__(self, v: Optional[float] = None):
        """Create a value node.

        Args:
            v: Optional initial value. If provided, the node starts as ready.
        """
        super().__init__()
        self.v: Optional[float] = None
        self.grad_v: Optional[float] = None
        if v is not None:
            self.v = v
            self.input_ready = True

    def receive_parent_value(self, v):
        """Store incoming value and mark this node ready."""
        self.v = v
        self.input_ready = True

    def set_grad_value(self, grad_v):
        """Set gradient explicitly.

        This is mainly a helper for manual experiments/tests.
        """
        self.grad_v = grad_v

    def _reset_local(self):
        self.v = None
        self.grad_v = None
        self.input_ready = False

    def forward(self):
        """Forward stored value to all children.

        Raises:
            Exception: If value is missing when forward is requested.
        """
        if self.v is None:
            raise Exception(
                "Forward not possible as no value set in this ValueNode"
            )
        for node in self.children:
            node.receive_parent_value(self.v)
            node.forward()

    def backward(self, grad_z):
        """Accumulate gradient and pass it to parents.

        A ``ValueNode`` can receive multiple downstream contributions;
        those are summed in ``grad_v``.
        """
        if self.grad_v is None:
            self.grad_v = grad_z
        else:
            self.grad_v += grad_z
        for node in self.parents:
            node.backward(grad_z)
