from math import sqrt

from computational_graph import CompGraph, MetaNode, ValueNode
from computational_graph.optimizers.abstract_optimizer import AbstractOptimizer


class RMSProp(AbstractOptimizer):
    def __init__(self, comp_graph: CompGraph, learning_rate: float, decay: float = 0.9, eps: float = 1e-8):
        super().__init__(comp_graph, learning_rate)
        self.decay = decay
        self.eps = eps
        self.velocity = {}

    def step(self):
        """
        Performs one optimization step by visiting and updating trainable nodes in a computational graph.

        The optimizer traverses the computational graph starting from its input nodes and applies the optimization
        logic to all trainable nodes. It uses momentum-based gradient descent to update the parameter values.
        """
        visited = set()

        def visit(node: MetaNode):
            if not isinstance(node, ValueNode) or not node.trainable or id(node) in visited:
                return
            visited.add(id(node))

            self.velocity[node] = self.decay * self.velocity.get(node, 0.0) + (1 - self.decay) * node.grad_v**2
            node.v -= (self.learning_rate / sqrt(self.velocity[node] + self.eps)) * node.grad_v

            for child in node.children:
                visit(child)

        for in_node in self.comp_graph.in_nodes:
            visit(in_node)

        self.comp_graph.zero_grad()
