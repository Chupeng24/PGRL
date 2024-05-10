import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

rng = np.random.default_rng()  # 构造一个默认位生成器(PCG64)

if __name__ == '__main__':
    uniform_weights = rng.dirichlet(alpha=(0.2, 0.2, 0.2), size=1000)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(uniform_weights[:, 0], uniform_weights[:, 1], uniform_weights[:, 2])

    ax.set_xlabel("x - makespan")
    ax.set_ylabel("y - total machine load")
    ax.set_zlabel("z - critical machine load")

    plt.show()