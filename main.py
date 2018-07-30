import numpy as np
from tree import DOO


def objective(x: float) -> float:
    """
    Global optimum on [0, 1] is 
    x = 0.86422 and f(x) = 0.975599
    """

    return 0.5 * (np.sin(13 * x) * np.sin(27 * x) + 1)


if __name__ == '__main__':
    tree = DOO(objective, delta=lambda h: 14 * 2**(-h))
    print(tree.search(150))