from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import total_ordering
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
from heapq import heappop, heappush
import networkx as nx
from tqdm import tqdm


@total_ordering
@dataclass
class Node:
    score: float
    key: float
    value: float
    interval: Tuple[float, float]
    depth: int
    label: int
    data: List[float]
    left: Optional[Node] = None
    right: Optional[Node] = None
    parent: Optional[Node] = None

    def __eq__(self, other) -> bool:
        return self.score == other.score

    def __lt__(self, other) -> bool:
        return self.score < other.score

    def __str__(self) -> str:
        return f"key: {self.key}, value: {self.value}, data: {self.data}"


class Tree(metaclass=ABCMeta):
    """
    Tree search methods for finding the maximum of objective function on [0, 1]
    """

    def __init__(self, objective: Callable[[float], float],
                 delta: Callable[[int, int, int, float], float],
                 eta: float = 0.0) -> None:
        self.nodes: List[Node] = []
        self.x_optimal: float = np.nan
        self.f_optimal: float = np.nan
        self.num = 1
        self.height = 1
        self.graph = nx.DiGraph()

        self.objective = objective
        self.delta = delta
        self.eta = eta

        value = self.objective(0.5)
        delta_h = self.delta(0, 1, 1, eta)
        self.root = Node(-(value + delta_h),
                         0.5, value, (0.0, 1.0), 0, 1, [value])
        self.nodes.append(self.root)
        self.graph.add_node(1, pos=(0.5, 0))

    def search(self, time_horizon: int) -> Dict[str, float]:
        for _ in tqdm(range(time_horizon)):
            node = self._select_node()
            self._update_optimum(node)
            self._expand(node)

        return {'x': self.x_optimal, 'obj': self.f_optimal}

    def _add_children(self, node: Node) -> None:
        """
        Split the corresponding cell into two intervals; for each interval,
        evaluate its mid-point and generate a child
        """

        depth_next = node.depth + 1

        node.left = self._add_child(node, 0.5 * (node.interval[0] + node.key),
                                    2 * node.label, depth_next,
                                    (node.interval[0], node.key))

        node.right = self._add_child(node, 0.5 * (node.key + node.interval[1]),
                                     2 * node.label + 1, depth_next,
                                     (node.key, node.interval[1]))

    def _add_child(self, node: Node, key: float, label: int,
                   depth_next: int,
                   interval: Tuple[float, float]) -> Node:
        value = self.objective(key)
        data = [value]
        delta_h = self.delta(depth_next, self.num, len(data), self.eta)
        node_new = Node(-(value + delta_h), key, value, interval, depth_next,
                        label, data)
        heappush(self.nodes, node_new)

        self.graph.add_node(label, pos=(key, -depth_next))
        self.graph.add_edge(root.label, label)
        node_new.parent = node

        return node_new

    @abstractmethod
    def _select_node(self) -> Node:
        """
        Select a node for expanding/sampling
        """

    @abstractmethod
    def _update_optimum(self, node: Node) -> None:
        """
        Update optimum if a better solution is found
        """

    @abstractmethod
    def _expand(self, node: Node) -> None:
        """
        Add children
        """

    @property
    def optimum(self) -> Tuple[float, float]:
        return (self.x_optimal, self.f_optimal)


class DOO(Tree):
    def _select_node(self) -> Node:
        return heappop(self.nodes)

    def _update_optimum(self, node: Node) -> None:
        if node.value > self.f_optimal:
            self.f_optimal = node.value
            self.x_optimal = node.key

    def _expand(self, node: Node) -> None:
        self._add_children(node)


class StoOO(Tree):
    def _select_node(self) -> Node:
        return heappop(self.nodes)

    def _update_optimum(self, node: Node) -> None:
        if node.depth >= self.height:
            self.f_optimal = node.value
            self.x_optimal = node.key
            self.height = node.depth

    def _collect_data(self, node: Node) -> None:
        self.num += 1
        sample = self.objective(node.key)
        node.data.append(sample)
        sample_size = len(node.data)
        node.value = (node.value * (sample_size - 1) + sample) / sample_size

        node.score = -(node.value + self.delta(node.depth + 1, self.num,
                                               sample_size, self.eta))

    def _expand(self, node: Node) -> None:
        self._collect_data(node)
        rate = (np.log(self.num * self.num / self.eta)
                if self.eta > 0 else np.inf)

        if 2 * self.delta(node.depth) ** 2 * len(node.data) >= rate:
            self._add_children(node)
        else:
            heappush(self.nodes, node)


class HOO(StoOO):
    def _select_node(self) -> Node:
        curr = self.root

        while not (curr.left is None or curr.right is None):
            curr = (curr.left
                    if curr.left.score < curr.right.score
                    else curr.right)

        return curr

    def _update_optimum(self, node: Node) -> None:
        self.f_optimal = node.value
        self.x_optimal = node.key

    def _expand(self, node: Node) -> None:
        self._collect_data(node)
        self._add_children(node)
        self._update_b_value(node)

    def _update_b_value(self, node: Node) -> None:
        if node is not None:
            size = len(node.left.data) + len(node.right.data)
            score = -(node.left.value * len(node.left.data) +
                      node.right.value * len(node.right.data)) / size
            score -= self.delta(node.depth, self.num, size, self.eta)
            node.score = max(score, min(
                node.left.score, node.right.score))
            self._update_b_value(node.parent)

    @property
    def optimum(self) -> Tuple[float, float]:
        return (self.x_optimal, self.f_optimal)
