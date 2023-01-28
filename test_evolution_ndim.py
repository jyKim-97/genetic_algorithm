import genalg.evolve as evolve
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import genalg.logger as logger


ndim = 20


def fpotential(args):
    # generalized Rosenbrock's function
    data = args[0]
    fros = np.sum(100 * (data[:-1] - data[1:])**2 + (data[:-1] - 1)**2)
    return -fros


if __name__ == "__main__":

    np.random.seed(2000)

    solver = evolve.EA(ndim, mu=3, num_select=5,
                    num_offspring=20, num_parent=50, use_multiprocess=False,
                    do_mutate=True, crossover_type="undx")
    solver.set_object_func(fpotential)
    pmin = np.ones(ndim) * (-10)
    pmax = np.ones(ndim) * 10
    solver.set_min_max(pmin, pmax)

    solver.check_setting()
    solver.random_initialization()
    # solver.load_history("./log")
    
    max_iter = 1000
    for n in tqdm(range(max_iter)):
        solver.next_generation()
        solver.print_log()

    # read log file
    log = logger.read_log("./log/log.txt")
    plt.figure(dpi=120, figsize=(4,4))
    plt.plot(np.average(log, axis=1), 'k')
    plt.savefig("./result.png")
    plt.show()