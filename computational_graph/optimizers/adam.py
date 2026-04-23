from math import sqrt

from computational_graph import CompGraph, MetaNode, ValueNode
from computational_graph.optimizers.abstract_optimizer import AbstractOptimizer


class Adam(AbstractOptimizer):
    def __init__(
        self, comp_graph: CompGraph, learning_rate: float, beta1: float = 0.9, beta2: float = 0.9, eps: float = 1e-8
    ):
        super().__init__(comp_graph, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.first_moment = {}
        self.second_moment = {}
        self.passes = 1

    def step(self):
        visited = set()

        def visit(node: MetaNode):
            if not isinstance(node, ValueNode) or not node.trainable or id(node) in visited:
                return
            visited.add(id(node))

            self.first_moment[node] = self.beta1 * self.first_moment.get(node, 0.0) + (1 - self.beta1) * node.grad_v
            self.second_moment[node] = (
                self.beta2 * self.second_moment.get(node, 0.0) + (1 - self.beta2) * node.grad_v**2
            )
            scaled_first_moment = self.first_moment[node] / (1 - self.beta1**self.passes)
            scaled_second_moment = self.second_moment[node] / (1 - self.beta2**self.passes)

            node.v -= self.learning_rate * scaled_first_moment / (sqrt(scaled_second_moment) + self.eps)

            for child in node.children:
                visit(child)

        for in_node in self.comp_graph.in_nodes:
            visit(in_node)

        self.comp_graph.zero_grad()
        self.passes += 1
