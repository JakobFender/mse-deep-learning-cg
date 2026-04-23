from __future__ import annotations

from abc import ABC, abstractmethod


class MetaNode(ABC):
    """Base abstraction for all graph nodes.

    Attributes:
        name (str): Human-readable identifier used in ``__repr__`` and debugging.
        parents (list[MetaNode]): Upstream nodes that feed values into this node.
        children (list[MetaNode]): Downstream nodes that receive values from this node.
        input_ready (bool): Whether all required inputs have been received.
    """

    def __init__(self, name: str = ""):
        self.name: str = name
        self.parents: list[MetaNode] = []
        self.children: list[MetaNode] = []
        self.input_ready: bool = False

    def add_child(self, node: MetaNode):
        """Register a downstream child node (one-directional).

        Note:
            Does not update ``node.parents``. Prefer ``connect_to`` for
            bidirectional linking.

        Args:
            node (MetaNode): The child node to register.
        """
        self.children.append(node)

    def add_parent(self, node: MetaNode):
        """Register an upstream parent node (one-directional).

        Note:
            Does not update ``node.children``. Prefer ``connect_to`` for
            bidirectional linking.

        Args:
            node (MetaNode): The parent node to register.
        """
        self.parents.append(node)

    def connect_to(self, node: MetaNode):
        """Connect this node to a downstream node (bidirectional).

        Appends ``node`` to ``self.children`` and ``self`` to ``node.parents``.

        Args:
            node (MetaNode): The downstream node to connect to.
        """
        self.children.append(node)
        node.parents.append(self)

    @abstractmethod
    def forward(self):
        """Compute local output and push it to children when ready."""
        pass

    @abstractmethod
    def _reset_local(self):
        """Clear node-local cached state in preparation for a new pass."""
        pass

    def reset_values(self):
        """Clear cached state and recursively reset all descendants.

        Calls ``_reset_local`` on this node, then calls ``reset_values``
        on each child in order.
        """
        self._reset_local()
        for node in self.children:
            node.reset_values()

    def _zero_local_grad(self):
        """Clears gradient in preparation for a new optimizer step."""
        pass

    def zero_grad(self):
        """Clear gradient and recursively reset all descendants.

        Calls ``zero_local_grad`` on this node, then calls ``zero_grad``
        on each child in order.
        """
        self._zero_local_grad()
        for node in self.children:
            node.zero_grad()

    @abstractmethod
    def backward(self, grad_z: float, batch_size: int = 1):
        """Propagate the gradient from a downstream node to upstream parents.

        Args:
            grad_z (float): Gradient of the loss with respect to this node's output.
            batch_size (int): Batch size to scale gradient
        """
        pass

    @abstractmethod
    def receive_parent_value(self, v: float):
        """Receive one upstream value from a parent node.

        Args:
            v (float): The value passed down from a parent node.
        """
        pass
