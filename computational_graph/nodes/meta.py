from __future__ import annotations

from abc import ABC, abstractmethod


class MetaNode(ABC):
    """Base abstraction for all graph nodes.

    A node keeps references to upstream ``parents`` and downstream
    ``children``. Implementations are responsible for tracking when inputs
    are ready (``input_ready``) and for propagating values/gradients.
    """

    def __init__(self):
        self.parents: list[MetaNode] = []
        self.children: list[MetaNode] = []
        self.input_ready: bool = False

    def add_child(self, node: MetaNode):
        """Register a downstream child node.

        This helper only updates the current node and does not update
        ``node.parents``. Prefer ``connect_to`` for bidirectional linking.
        """
        self.children.append(node)

    def add_parent(self, node: MetaNode):
        """Register an upstream parent node.

        This helper only updates the current node and does not update
        ``node.children``. Prefer ``connect_to`` for bidirectional linking.
        """
        self.parents.append(node)

    def connect_to(self, node: MetaNode):
        """Connect this node to another node.

        The connection is created in both directions:
        ``self -> node`` and ``node <- self``.
        """
        self.children.append(node)
        node.parents.append(self)

    @abstractmethod
    def forward(self):
        """Compute local output and push it to children when ready."""
        pass

    @abstractmethod
    def _reset_local(self):
        """Clear node-local cached state for a new pass."""
        pass

    def reset_values(self):
        """Clear cached forward/backward state and recursively reset descendants."""
        self._reset_local()
        for node in self.children:
            node.reset_values()

    @abstractmethod
    def backward(self, grad_z):
        """Propagate gradient from downstream to upstream parents."""
        pass

    @abstractmethod
    def receive_parent_value(self, v):
        """Receive one upstream value from a parent node."""
        pass
