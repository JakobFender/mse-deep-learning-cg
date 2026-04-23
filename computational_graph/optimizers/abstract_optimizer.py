from abc import ABC, abstractmethod

from computational_graph import CompGraph


class AbstractOptimizer(ABC):
    def __init__(self, comp_graph: CompGraph, learning_rate: float):
        """
        Initializes the class with a computational graph and learning rate.

        Args:
            comp_graph: Computational graph to be used for executing operations
                or optimizations.
            learning_rate: Learning rate value to be used in optimization processes.
        """
        self.comp_graph = comp_graph
        self.learning_rate = learning_rate

    @abstractmethod
    def step(self):
        raise NotImplementedError
