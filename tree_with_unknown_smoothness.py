from __future__ import annotations

import numpy as np
from heapq import heappop, heappush
from typing import Callable, List

from tree import Node, SOOTree


class SOO(SOOTree):
    def __init__(self, max_iters: int, num_of_children: int,
                 h_max: Callable[[int], int]) -> None:

        Node.delta = lambda node: 0.0
        super().__init__(max_iters, num_of_children, h_max)

    def _select_node(self, queue: List[Node],
                     objective: Callable[[float], float]) -> Node:
        return heappop(queue)

    def _expand(self, node: Node, objective: Callable[[float], float]) -> bool:
        self._add_children(node, objective)

        return True

    def _update_optimum(self, node: Node) -> None:
        self.x_optimal = node.mid
        self.f_optimal = node.value


class StoSOO(SOO):
    def __init__(self, max_iters: int, num_of_children: int,
                 h_max: Callable[[int], int], sample_per_node: int,
                 eta: float) -> None:
        assert sample_per_node > 1, "Sample per node must be greater 1."
        self.sample_per_node = sample_per_node

        rate = np.sqrt(np.log(max_iters * max_iters / eta) * 0.5)

        def delta(node: Node, rate: float = rate) -> float:
            return rate / np.sqrt(node.size)

        Node.delta = delta
        super().__init__(max_iters, num_of_children, h_max)

    def _expand(self, node: Node, objective: Callable[[float], float]) -> bool:

        if node.size < self.sample_per_node:
            node.evaluate(objective)
            heappush(self.nodes[node.depth], node)
            self.size += 1

            return False
        else:
            self._add_children(node, objective)

            return True
