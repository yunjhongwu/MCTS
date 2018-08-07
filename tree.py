from __future__ import annotations

import abc
from functools import total_ordering
from dataclasses import dataclass, field
from heapq import heappush
import numpy as np
from tqdm import tqdm
from typing import Callable, ClassVar, Dict, List, Optional, Tuple


@total_ordering
@dataclass
class Node:
    tree: ClassVar[BaseTree] 
    delta: ClassVar[Callable[..., float]] = lambda dummy: 0.0

    lower: float
    upper: float
    depth: int
    label: int
    data: List[Tuple[float, float]] = field(default_factory=list)
    size: int = 0
    value: float = np.inf
    score: float = np.inf
    parent: Optional[Node] = None
    children: List[Node] = field(default_factory=list)

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
                 random_state: bool = False):
        """
        Calculate score
        """
        key = np.random.uniform(self.lower,
                                self.upper) if random_state else self.mid
        sample = objective(key)
        self.collect(key, sample)

    def collect(self, key: float, sample: float):
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


class BaseTree(metaclass=abc.ABCMeta):
    """
    Tree search methods for finding the maximum of objective function on [0, 1]
    """

    def _initialize(self) -> None:
        self.size = 0
        self.height = 1
        self.root = Node(0.0, 1.0, 0, 1)
        self.x_optimal: float = np.nan
        self.f_optimal: float = -np.inf

    @abc.abstractmethod
    def _select_node(self, queue: List[Node],
                     objective: Callable[[float], float]) -> Node:
        """
        Select a node for expanding/sampling
        """

    @abc.abstractmethod
    def _expand(self, node: Node, objective: Callable[[float], float]) -> None:
        """
        Add children
        """

    @abc.abstractmethod
    def _update_optimum(self, node: Node) -> None:
        """
        Update optimum if a better solution is found
        """


class OOTree(BaseTree):
    def __init__(self,
                 max_iters: int,
                 num_of_children: int,
                 delta: Callable[[Node], float]) -> None:
        assert num_of_children > 1, "Number of children must be greater 1."

        Node.delta = delta
        Node.tree = self
        self.max_iters = max_iters
        self.num_of_children = num_of_children

    def search(self, objective: Callable[[float], float]) -> Dict[str, float]:
        self._initialize()

        self.nodes: List[Node] = [self.root]

        progressbar = tqdm(total=self.max_iters)

        while self.size < self.max_iters:
            node = self._select_node(self.nodes, objective)
            self._expand(node, objective)
            self._update_optimum(node)

            progressbar.n = min(self.size, self.max_iters)
            progressbar.refresh()

        progressbar.close()

        return {'x': self.x_optimal, 'obj': self.f_optimal}

    def _add_children(self, node: Node) -> None:
        """
        Split the corresponding cell into two intervals; for each interval,
        generate a child
        """

        depth_next = node.depth + 1
        width = (node.upper - node.lower) / self.num_of_children
        for i in range(self.num_of_children):
            label = node.label * self.num_of_children + i
            child = Node(node.lower + i * width, node.lower + (i + 1) * width,
                         depth_next, label)
            node.add_child(child)
            heappush(self.nodes, child)


class SOOTree(BaseTree):
    def __init__(self, max_iters: int, num_of_children: int,
                 h_max: Callable[[int], int]) -> None:
        Node.tree = self
        self.h_max = h_max
        self.num_of_children = num_of_children
        self.max_iters = max_iters

    def search(self, objective: Callable[[float], float]) -> Dict[str, float]:
        self._initialize()
        self.root.evaluate(objective)
        self.nodes = [[self.root]]

        progressbar = tqdm()
        scope = 0
        height_bound = 1
        while scope < height_bound and self.size < self.max_iters:
            v_max = -np.inf
            for h in range(0, height_bound):
                if len(self.nodes[h]) > 0:
                    node = self._select_node(self.nodes[h], objective)
                    if node.score >= v_max:
                        v_max = node.value
                        self._expand(node, objective)
                        self._update_optimum(node)

                        height_bound = min(self.height, self.h_max(self.size))

                else:
                    scope = h + 1

            progressbar.n = min(self.size, self.max_iters)
            progressbar.refresh()

        progressbar.close()

        return {'x': self.x_optimal, 'obj': self.f_optimal}

    def _add_children(self, node: Node,
                      objective: Callable[[float], float]) -> None:
        """
        Split the corresponding cell into two intervals; for each interval,
        generate and evaluate a child
        """

        depth_next = node.depth + 1
        if depth_next >= len(self.nodes):
            self.nodes.append([])
            self.height += 1

        width = (node.upper - node.lower) / self.num_of_children
        for i in range(self.num_of_children):
            label = node.label * self.num_of_children + i + 1
            child = Node(node.lower + i * width, node.lower + (i + 1) * width,
                         depth_next, label)
            child.evaluate(objective)
            node.add_child(child)
            heappush(self.nodes[depth_next], child)
