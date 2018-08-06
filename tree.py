from __future__ import annotations

import abc
from functools import total_ordering
from heapq import heappush
import numpy as np
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Tuple


@total_ordering
class Node:
    delta: Optional[Callable[..., float]] = lambda dummy: 0.0

    def __init__(self, lower: float, upper: float, depth: int,
                 label: int) -> None:
        self.lower = lower
        self.upper = upper
        self.depth = depth
        self.label = label
        self.data: List[Tuple[float, float]] = list()
        self.size: int = 0
        self.value: float = np.inf
        self.score: float = np.inf
        self.parent: Optional[Node] = None
        self.children: List[Node] = []

    def __eq__(self, other) -> bool:
        return self.score == other.score

    def __lt__(self, other) -> bool:
        return self.score > other.score

    def __str__(self) -> str:
        return (f"Node {self.label}: depth: {self.depth}, "
                f"interval: ({self.lower}, {self.upper}), "
                f"value: {self.value}, data: {self.data}")

    def evaluate(self,
                 objective: Callable[[float], float],
                 update_time: int,
                 random_state: bool = False):
        """
        Calculate score
        """
        key = np.random.uniform(self.lower,
                                self.upper) if random_state else self.mid
        sample = objective(key)
        self.collect(key, sample, update_time)

    def collect(self, key: float, sample: float, update_time: int):
        self.update_time = update_time
        self.value = ((self.value * self.size + sample) / (self.size + 1)
                      if self.size > 0 else sample)
        self.data.append((key, sample))
        self.size += 1
        self.update_score()

    def update_score(self):
        self.score = self.value + self.delta()

    def add_child(self, node: Node):
        node.parent = self
        self.children.append(node)

    @property
    def mid(self):
        return (self.lower + self.upper) * 0.5


class Tree(metaclass=abc.ABCMeta):
    """
    Tree search methods for finding the maximum of objective function on [0, 1]
    """

    def __init__(self,
                 max_iters: int,
                 num_of_children: int,
                 delta: Callable[[Node], float],
                 eta: float = 0.0) -> None:
        assert num_of_children > 1, "Number of children must be greater 1."
        Node.delta = delta
        self.max_iters = max_iters
        self.num_of_children = num_of_children
        self.eta = eta

    def search(self, objective: Callable[[float], float]) -> Dict[str, float]:
        self._initialize()
        progressbar = tqdm(total=self.max_iters)

        while self.size < self.max_iters:
            node = self._select_node(objective)
            self._expand(node)
            self._update_optimum(node)

            progressbar.n = min(self.size, self.max_iters)
            progressbar.refresh()

        progressbar.close()

        return {'x': self.x_optimal, 'obj': self.f_optimal}

    def _initialize(self) -> None:
        self.size = 0
        self.height = 1
        self.nodes: List[Node] = []
        self.root = Node(0.0, 1.0, 0, 0)
        self.nodes.append(self.root)
        self.x_optimal: float = np.nan
        self.f_optimal: float = -np.inf

    def _add_children(self, node: Node) -> None:
        """
        Split the corresponding cell into two intervals; for each interval,
        evaluate generate a child
        """

        depth_next = node.depth + 1
        width = (node.upper - node.lower) / self.num_of_children
        for i in range(self.num_of_children):
            label = node.label * self.num_of_children + i + 1
            child = Node(node.lower + i * width, node.lower + (i + 1) * width,
                         depth_next, label)
            node.add_child(child)
            heappush(self.nodes, child)

    @abc.abstractmethod
    def _select_node(self, objective: Callable[[float], float]) -> Node:
        """
        Select a node for expanding/sampling
        """

    @abc.abstractmethod
    def _expand(self, node: Node) -> None:
        """
        Add children
        """

    @abc.abstractmethod
    def _update_optimum(self, node: Node) -> None:
        """
        Update optimum if a better solution is found
        """
