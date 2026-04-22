from __future__ import annotations

from .nodes.value import ValueNode


class CompGraph:
    """A simple computational graph entry point.

    Manages input and output ``ValueNode`` objects and orchestrates full
    forward and backward passes across the graph.

    Attributes:
        in_nodes (list[ValueNode]): Ordered input placeholder nodes.
        out_nodes (list[ValueNode]): Ordered output nodes.
        forwarded (bool): Whether a successful forward pass has been run.
    """

    def __init__(self, in_nodes: list[ValueNode], out_nodes: list[ValueNode]):
        """Create a graph wrapper around input and output value nodes.

        Args:
            in_nodes (list[ValueNode]): Ordered list of external input placeholders.
            out_nodes (list[ValueNode]): Ordered list of nodes considered graph outputs.

        Raises:
            Exception: If any node in ``in_nodes`` or ``out_nodes`` is not a
                ``ValueNode``.
        """
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.forwarded = False
        self.validate_graph()

    def validate_graph(self):
        """Validate that all boundary nodes are ``ValueNode`` instances.

        Raises:
            Exception: If any node in ``in_nodes`` or ``out_nodes`` is not a
                ``ValueNode``.
        """
        for node in self.in_nodes:
            if not isinstance(node, ValueNode):
                raise Exception("Input node of CompGraph is not a ValueNode")
        for node in self.out_nodes:
            if not isinstance(node, ValueNode):
                raise Exception("Output node of CompGraph is not a ValueNode")

    def forward(self, input_values: list[float]):
        """Run a forward pass, feeding values into input nodes.

        Args:
            input_values (list[float]): Scalar values fed to ``in_nodes`` in the
                same order.

        Raises:
            Exception: If the length of ``input_values`` does not match the number
                of input nodes.
        """
        if len(input_values) != len(self.in_nodes):
            raise Exception("Can't forward: number of input differs to number of input nodes")
        for i, in_node in enumerate(self.in_nodes):
            in_node.receive_parent_value(input_values[i])
            in_node.forward()
        self.forwarded = True

    def backward(self):
        """Run a backward pass from all output nodes with a unit upstream gradient.

        Raises:
            Exception: If called before at least one successful forward pass.
        """
        if not self.forwarded:
            raise Exception("Can't backward, you need to call forward first")
        for node in self.out_nodes:
            node.backward(1.0)

    def reset_values(self):
        """Reset all cached values recursively so a new pass can be run.

        Resets the entire subgraph reachable from ``in_nodes`` and clears the
        ``forwarded`` flag.
        """
        for node in self.in_nodes:
            node.reset_values()
        self.forwarded = False

    def __repr__(self) -> str:
        lines = []
        visited = set()

        def visit(node):
            if id(node) in visited:
                return
            visited.add(id(node))

            lines.append(str(node))
            for i, child in enumerate(node.children):
                visit(child)

        lines.append(" --- Input Nodes --- ")
        for input_node in self.in_nodes:
            lines.append(str(input_node))
            visited.add(id(input_node))
        lines.append("")

        for output_node in self.out_nodes:
            visited.add(id(output_node))

        lines.append(" --- Intermediate Notes --- ")
        for node in self.in_nodes:
            for child in node.children:
                visit(child)
        lines.append("")

        lines.append(" --- Output Notes --- ")
        for output_node in self.out_nodes:
            lines.append(str(output_node))
        lines.append("")

        return "\n".join(lines)
