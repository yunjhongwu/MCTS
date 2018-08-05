from __future__ import annotations

import abc
from functools import total_ordering
from heapq import heappush
import numpy as np
from tqdm import tqdm
from typing import Callable, Dict, List, Optional


@total_ordering
class Node:
    delta: Optional[Callable[..., float]] = None

    def __init__(self, lower: float, upper: float, depth: int,
                 label: int) -> None:
        self.lower = lower
        self.upper = upper
        self.depth = depth
        self.label = label
        self.data: List[float] = list()
        self.size: int = 0
        self.value: float = np.inf
        self.score: float = np.inf
        self.parent: Optional[Node] = None
        self._left: Optional[Node] = None
        self._right: Optional[Node] = None

    def __eq__(self, other) -> bool:
        return self.score == other.score

    def __lt__(self, other) -> bool:
        return self.score > other.score

    def __str__(self) -> str:
        return (f"Node {self.label}: depth: {self.depth}, "
                f"interval: ({self.lower}, {self.upper}), "
                f"value: {self.value}, data: {self.data}")

    def collect(self, sample: float, update_time: int):
        """
        Calculate score
        """
        self.update_time = update_time
        self.value = ((self.value * self.size + sample) / (self.size + 1)
                      if self.size > 0 else sample)
        self.data.append(sample)
        self.size += 1
        self.update_score()

    def sample(self):
        return np.random.uniform(self.lower, self.upper)

    def update_score(self):
        self.score = self.value + self.delta()

    def set_value(self, value: float, size: int):
        self.value = value
        self.update_score()

    @property
    def mid(self):
        return (self.lower + self.upper) * 0.5

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, node: Node):
        node.parent = self
        self._left = node

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, node: Node):
        node.parent = self
        self._right = node


class Tree(metaclass=abc.ABCMeta):
    """
    Tree search methods for finding the maximum of objective function on [0, 1]
    """

    def __init__(self,
                 max_iters: int,
                 delta: Callable[[Node], float],
                 eta: float = 0.0) -> None:
        Node.delta = delta




        self.max_iters = max_iters
        self.eta = eta

    def search(self, objective: Callable[[float], float]) -> Dict[str, float]:
        self.size = 0
        self.height = 1
        self.nodes: List[Node] = []
        self.root = Node(0.0, 1.0, 0, 1)
        self.nodes.append(self.root)
        self.x_optimal: float = np.nan
        self.f_optimal: float = -np.inf

        progressbar = tqdm(total=self.max_iters)

        while self.size < self.max_iters:
            node = self._select_node(objective)
            self._expand(node)
            self._update_optimum(node)

            progressbar.n = min(self.size, self.max_iters)
            progressbar.refresh()

        progressbar.close()

        return {'x': self.x_optimal, 'obj': self.f_optimal}

    def _add_children(self, node: Node) -> None:
        """
        Split the corresponding cell into two intervals; for each interval,
        evaluate generate a child
        """

        depth_next = node.depth + 1
        node.left = Node(node.lower, node.mid, depth_next, 2 * node.label)
        node.right = Node(node.mid, node.upper, depth_next, 2 * node.label + 1)
        heappush(self.nodes, node.left)
        heappush(self.nodes, node.right)

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
