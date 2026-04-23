from computational_graph import CompGraph, MetaNode, ValueNode
from computational_graph.optimizers.abstract_optimizer import AbstractOptimizer


class GradientDescent(AbstractOptimizer):
    def __init__(self, comp_graph: CompGraph, learning_rate: float, momentum: float = 0.0):
        super().__init__(comp_graph, learning_rate)
        self.momentum = momentum
        self.velocity = {}

    def step(self):
        """
        Updates the values of trainable `MetaNodes` based.

        The `step` method performs a depth-first traversal of the computational graph, starting
        from the input nodes. It modifies nodes of the type `ValueNode` that are trainable by
        adjusting their value using the gradient of the node and the specified learning rate. To
        prevent duplicate visits, a set of visited nodes is maintained.
        """
        visited = set()

        def visit(node: MetaNode):
            if not isinstance(node, ValueNode) or not node.trainable or id(node) in visited:
                return
            visited.add(id(node))

            self.velocity[node] = self.momentum * self.velocity.get(node, 0.0) + node.grad_v
            node.v -= self.learning_rate * self.velocity[node]

            for child in node.children:
                visit(child)

        for in_node in self.comp_graph.in_nodes:
            visit(in_node)

        self.comp_graph.zero_grad()
