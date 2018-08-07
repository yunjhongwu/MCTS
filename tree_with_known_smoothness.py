from __future__ import annotations

import numpy as np
from typing import Callable, List, Optional
from heapq import heappop, heappush

from tree import Node, OOTree


class DOO(OOTree):
    def _select_node(self, queue: List[Node],
                     objective: Callable[[float], float]) -> Node:
        node = heappop(queue)

        while np.isinf(node.value):
            self.size += 1
            node.evaluate(objective)
            heappush(self.nodes, node)
            node = heappop(self.nodes)

        return node

    def _expand(self, node: Node, objective: Callable[[float], float]) -> None:
        self._add_children(node)

    def _update_optimum(self, node: Node) -> None:
        if node.value > self.f_optimal:
            self.f_optimal = node.value
            self.x_optimal = node.mid


class StoOO(OOTree):
    def __init__(self, max_iters: int, num_of_children: int,
                 delta: Callable[[Node], float], eta: float) -> None:
        assert eta > 0, "Eta must be strictly positive."
        self.delta = delta
        self.rate = np.sqrt(np.log(max_iters * max_iters / eta) * 0.5)

        def upper_bound(node: Node, rate: float = self.rate) -> float:
            return delta(node) + rate / np.sqrt(node.size)

        super().__init__(max_iters, num_of_children, upper_bound)

    def _select_node(self, queue: List[Node],
                     objective: Callable[[float], float]) -> Node:
        self.size += 1
        node = heappop(queue)
        node.evaluate(objective)

        return node

    def _expand(self, node: Node, objective: Callable[[float], float]) -> None:
        if 2 * self.delta(node)**2 * node.size >= self.rate:
            self._add_children(node)
        else:
            heappush(self.nodes, node)

    def _update_optimum(self, node: Node) -> None:
        if node.depth >= self.height:
            self.f_optimal = node.value
            self.x_optimal = node.mid
            self.height = node.depth


class HOO(OOTree):
    def __init__(self, max_iters: int, num_of_children: int,
                 delta: Callable[[Node], float], eta: float) -> None:
        assert eta > 0, "Eta must be strictly positive."
        self.non_leaves: List[Node] = []
        self.delta = delta
        self.rate = np.sqrt(np.log(max_iters * max_iters / eta) * 0.5)

        def upper_bound(node: Node, rate: float = self.rate) -> float:
            return delta(node) + np.sqrt(
                2 * np.log(Node.tree.size) / node.size)

        super().__init__(max_iters, num_of_children, upper_bound)

    def _select_node(self, queue: List[Node],
                     objective: Callable[[float], float]) -> Node:
        curr = self.root

        while len(curr.children) > 0:
            idx = np.argmin(curr.children)
            curr = curr.children[idx]

        self.size += 1
        curr.evaluate(objective, random_state=True)

        return curr

    def _expand(self, node: Node, objective: Callable[[float], float]) -> None:
        self._add_children(node)
        self.non_leaves.append(node)

        while len(node.data) > 0:
            key, sample = node.data.pop()
            idx = int((key - node.lower) / (node.upper - node.lower) *
                      self.num_of_children)
            node.children[idx].collect(key, sample)
           
        self._update_b_value(node)

    def _update_optimum(self, node: Node) -> None:
        random_node = np.random.choice(self.non_leaves)
        self.f_optimal = random_node.value
        self.x_optimal = random_node.mid

    def _update_b_value(self, node: Optional[Node]) -> None:
        if node is not None:
            size = sum(child.size for child in node.children)
            total = sum(child.size * child.value for child in node.children
                        if child.size > 0)

            value = total / size
            node.value = value
            node.size = size

            node.score = min(
                value + np.sqrt(2 * np.log(self.size) / size) +
                self.delta(node), max(child.score for child in node.children))
            self._update_b_value(node.parent)
