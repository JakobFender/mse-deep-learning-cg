from __future__ import annotations

from .nodes.value import ValueNode


class CompGraph:
    """A simple computational graph entry point.

    It manages input/output ValueNode objects and orchestrates full
    forward/backward passes.
    """

    def __init__(self, in_nodes: list[ValueNode], out_nodes: list[ValueNode]):
        """Create a graph wrapper around input and output value nodes.

        Args:
            in_nodes: Ordered list of external input placeholders.
            out_nodes: Ordered list of nodes considered graph outputs.
        """
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.forwarded = False
        self.validate_graph()

    def validate_graph(self):
        """Validate that graph boundaries are made of ValueNode objects."""
        for node in self.in_nodes:
            if not isinstance(node, ValueNode):
                raise Exception("Input node of CompGraph is not a ValueNode")
        for node in self.out_nodes:
            if not isinstance(node, ValueNode):
                raise Exception("Output node of CompGraph is not a ValueNode")

    def forward(self, input_values: list[float]):
        """Run a forward pass from graph inputs to outputs.

        Args:
            input_values: Scalar values fed to input nodes in the same
                order as ``in_nodes``.

        Raises:
            Exception: If number of provided values does not match graph
                input count.
        """
        if len(input_values) != len(self.in_nodes):
            raise Exception(
                "Can't forward: number of input differs to number "
                "of input nodes"
            )
        for i, in_node in enumerate(self.in_nodes):
            in_node.receive_parent_value(input_values[i])
            in_node.forward()
        self.forwarded = True

    def backward(self):
        """Run backward pass from graph outputs with unit upstream gradient.

        Raises:
            Exception: If called before at least one successful forward pass.
        """
        if not self.forwarded:
            raise Exception("Can't backward, you need to call forward first")
        for node in self.out_nodes:
            node.backward(1.0)

    def reset_values(self):
        """Reset cached values recursively so a new pass can be executed."""
        for node in self.in_nodes:
            node.reset_values()
        self.forwarded = False
