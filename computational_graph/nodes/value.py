from __future__ import annotations

from typing import Optional

from .meta import MetaNode


class ValueNode(MetaNode):
    """A node that stores a scalar value and its accumulated gradient.

    Attributes:
        v (float | None): The scalar value held by this node. ``None`` until set.
        grad_v (float | None): Accumulated gradient. ``None`` until a backward
            pass contributes at least one gradient.
    """

    def __init__(self, name: str = "", v: Optional[float] = None, trainable=False):
        """Create a value node.

        Args:
            name (str): Human-readable identifier for this node.
            v (float, optional): Initial scalar value. When provided, ``input_ready``
                is set to ``True`` immediately.
        """
        super().__init__(name)
        self.v: Optional[float] = None
        self.grad_v: Optional[float] = None
        if v is not None:
            self.v = v
            self.input_ready = True
        self.trainable = trainable

    def receive_parent_value(self, v: float):
        """Store an incoming value from a parent node and mark this node ready.

        Args:
            v (float): The scalar value forwarded by the upstream parent.
        """
        self.v = v
        self.input_ready = True

    def set_grad_value(self, grad_v: float):
        """Overwrite the gradient directly.

        This is mainly a helper for manual experiments and tests.

        Args:
            grad_v (float): The gradient value to store.
        """
        self.grad_v = grad_v

    def _reset_local(self):
        """Clear the stored value, gradient, and readiness flag."""
        self.v = None
        self.grad_v = None
        self.input_ready = False

    def forward(self):
        """Forward the stored value to all children.

        Raises:
            RuntimeError: If ``v`` is ``None`` when ``forward`` is called.
        """
        if self.v is None:
            raise RuntimeError("Forward not possible as no value set in this ValueNode")
        for node in self.children:
            node.receive_parent_value(self.v)
            node.forward()

    def backward(self, grad_z: float):
        """Accumulate an incoming gradient and propagate it to all parents.

        A ``ValueNode`` may receive gradient contributions from multiple
        downstream nodes; these are summed into ``grad_v``.

        Args:
            grad_z (float): Gradient of the loss with respect to this node's value.
        """
        if self.grad_v is None:
            self.grad_v = grad_z
        else:
            self.grad_v += grad_z
        for node in self.parents:
            node.backward(grad_z)

    def __repr__(self) -> str:
        return f"ValueNode(name={self.name}, v={self.v}, grad={self.grad_v})"
