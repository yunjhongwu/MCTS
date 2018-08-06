import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Callable, Dict, List, Tuple

from tree import Tree, Node


def visualize(tree: Tree, fun: Callable[[float], float]) -> None:
    fig, ax = plt.subplots(
        2, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [3, 1]})
    fig.tight_layout()

    graph: nx.DiGraph = nx.DiGraph()
    pos: Dict[int, Tuple[float, float]] = {}
    queue: List[Node] = [tree.root]
    values: List[Tuple[float, float]] = []

    while len(queue):
        node = queue.pop()

        values.extend(node.data)
        pos[node.label] = (node.mid, -node.depth)
        for child in node.children:
            if child is not None and np.isfinite(child.value):
                queue.append(child)
                graph.add_edge(node.label, child.label)

    nx.draw(graph, pos, node_size=10, with_labels=False, arrows=True, ax=ax[0])
    ax[0].set_xlim(0, 1)

    x = np.linspace(0, 1, 1000)
    ax[1].plot(x, fun(x))
    ax[1].scatter(*zip(*values), s=10, color='r')
    ax[1].set_xlim(0, 1)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].grid()
