from __future__ import annotations

from .meta import MetaNode
from .value import ValueNode


class AddNode(MetaNode):
    """An addition node that accepts a variable number of arguments: z = sum(x_i).

    Attributes:
        inputs (list[float]): Values received from parent nodes in the current pass.
    """

    def __init__(self, in_nodes: list[ValueNode], out_node: ValueNode):
        """Create an addition operator with an arbitrary number of inputs.

        Args:
            in_nodes (list[ValueNode]): Input addend nodes.
            out_node (ValueNode): Output node that receives the sum.
        """
        super().__init__()
        for node in in_nodes:
            node.connect_to(self)
        self.connect_to(out_node)
        self.inputs: list[float] = []

    def receive_parent_value(self, v: float):
        """Append one addend and mark the node ready when all inputs have arrived.

        Args:
            v (float): Scalar value forwarded by a parent node.

        Raises:
            Exception: If more values are received than there are parent nodes.
        """
        self.inputs.append(v)
        if len(self.inputs) == len(self.parents):
            self.input_ready = True
        elif len(self.inputs) > len(self.parents):
            raise Exception("All inputs are already set")

    def _reset_local(self):
        """Clear the collected input values and readiness flag."""
        self.inputs = []
        self.input_ready = False

    def forward(self):
        """Compute the sum of all inputs and push the result to children."""
        if self.input_ready:
            s = sum(self.inputs)
            for node in self.children:
                node.receive_parent_value(s)
                node.forward()

    def backward(self, grad_z: float):
        """Distribute the upstream gradient equally to all parent nodes.

        The gradient of a sum with respect to each addend is 1, so ``grad_z``
        is passed unchanged to every parent.

        Args:
            grad_z (float): Gradient of the loss with respect to this node's output.
        """
        grad_x = 1.0 * grad_z
        for node in self.parents:
            node.backward(grad_x)

    def __repr__(self) -> str:
        arguments = [f"in{i}={p.name}" for i, p in enumerate(self.parents)]
        return f"AddNode({', '.join(arguments)}, out={self.children[0].name})"
