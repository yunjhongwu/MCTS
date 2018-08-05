from __future__ import annotations

import numpy as np
from typing import Callable, Optional
from heapq import heappop, heappush

from tree import Node, Tree


class DOO(Tree):
    def _select_node(self, objective: Callable[[float], float]) -> Node:
        node = heappop(self.nodes)

        while np.isinf(node.value):
            self.size += 1
            node.collect(objective(node.mid), self.size)
            heappush(self.nodes, node)
            node = heappop(self.nodes)

        return node

    def _expand(self, node: Node) -> None:
        self._add_children(node)

    def _update_optimum(self, node: Node) -> None:
        if node.value > self.f_optimal:
            self.f_optimal = node.value
            self.x_optimal = node.mid


class StoOO(Tree):
    def __init__(self, max_iters: int, delta: Callable[[Node], float],
                 eta: float) -> None:
        assert eta > 0, "Eta must be strictly positive."
        self.delta = delta
        self.rate = np.sqrt(np.log(max_iters * max_iters / eta) * 0.5)

        def upper_bound(node: Node, rate: float = self.rate) -> float:
            return delta(node) + rate / np.sqrt(len(node.data))

        super().__init__(max_iters, upper_bound, eta)

    def _select_node(self, objective: Callable[[float], float]) -> Node:
        self.size += 1
        node = heappop(self.nodes)
        node.collect(objective(node.mid), self.size)

        return node

    def _expand(self, node: Node) -> None:
        if 2 * self.delta(node)**2 * len(node.data) >= self.rate:
            self._add_children(node)
        else:
            heappush(self.nodes, node)

    def _update_optimum(self, node: Node) -> None:
        if node.depth >= self.height:
            self.f_optimal = node.value
            self.x_optimal = node.mid
            self.height = node.depth


class HOO(StoOO):
    def __init__(self, max_iters: int, delta: Callable[[Node], float],
                 eta: float) -> None:
        assert eta > 0, "Eta must be strictly positive."
        self.delta = delta
        self.rate = np.sqrt(np.log(max_iters * max_iters / eta) * 0.5)

        def upper_bound(node: Node, rate: float = self.rate) -> float:
            return delta(node) + np.sqrt(
                2 * np.log(node.update_time) / len(node.data))

        super().__init__(max_iters, upper_bound, eta)

    def _select_node(self, objective: Callable[[float], float]) -> Node:
        curr = self.root

        while not (curr.left is None or curr.right is None):
            curr = (curr.left
                    if curr.left.score > curr.right.score else curr.right)

        self.size += 1
        curr.collect(objective(curr.sample()), self.size)
        return curr

    def _expand(self, node: Node) -> None:
        self._add_children(node)
        while len(node.data) > 0:
            obs = node.data.pop()
            (node.right
             if obs > node.mid else node.left).collect(obs, self.size)

        self._update_b_value(node)

    def _update_optimum(self, node: Node) -> None:
        self.f_optimal = node.value
        self.x_optimal = node.mid

    def _update_b_value(self, node: Optional[Node]) -> None:
        if node is not None:

            size = node.left.size + node.right.size
            left_sum = (node.left.value * node.left.size
                        if node.left.size > 0 else 0)
            right_sum = (node.right.value * node.right.size
                         if node.right.size > 0 else 0)

            value = (left_sum + right_sum) / size
            node.value = value
            node.size = size

            node.score = min(
                value + np.sqrt(2 * np.log(node.update_time) / size),
                max(node.left.score, node.right.score))
            self._update_b_value(node.parent)
