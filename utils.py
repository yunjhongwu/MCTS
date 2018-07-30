import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Callable

from tree import Tree


def visualize(tree: Tree, fun: Callable[[float], float]) -> None:
    fig, ax = plt.subplots(
        2, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [3, 1]})
    fig.tight_layout()
    pos = nx.get_node_attributes(tree.graph, "pos")
    nx.draw(tree.graph, pos, node_size=10,
            with_labels=False, arrows=True, ax=ax[0])
    ax[0].set_xlim(0, 1)

    queue = [tree.root]
    keys = []
    values = []
    while queue:
        curr = queue.pop()
        keys.append(curr.key)
        values.append(curr.value)
        if curr.left is not None:
            queue.append(curr.right)
        if curr.right is not None:
            queue.append(curr.left)
    
    x = np.linspace(0, 1, 1000)
    ax[1].plot(x, fun(x))
    ax[1].scatter(keys, values, s=10, color='r')
    ax[1].set_xlim(0, 1)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].grid()
