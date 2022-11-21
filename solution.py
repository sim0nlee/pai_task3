import numpy as np
from scipy.stats import norm
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

domain = np.array([[0, 5]])
SAFETY_THRESHOLD = 1.2
SEED = 0

UCB_BETA = 2.0

""" Solution """


class BO_algo():

    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """

        # TODO: enter your code here
        self.prev_evals = []  # previous evaluations are stored as a list of tuples of the form (x, f(x), v(x))

        self.f = GaussianProcessRegressor(
            kernel=ConstantKernel(0.5) * Matern(length_scale=0.5, nu=2.5),
            alpha=0.15 ** 2,
            optimizer=None,
            normalize_y=True
        )  # Gaussian Process modelling the accuracy mapping

        self.v = GaussianProcessRegressor(
            kernel=ConstantKernel(1.5) + ConstantKernel(np.sqrt(2)) * Matern(length_scale=0.5, nu=2.5),
            alpha=0.0001 ** 2,
            optimizer=None,
            normalize_y=True
        )  # Gaussian Process modelling the speed of evaluation of the accuracy mapping for a given hyperparameter

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # return a random point if this is the first evaluation, if not call the optimization procedure
        return np.array([[np.random.uniform(0, 5)]]) if not self.prev_evals else self.optimize_acquisition_function()

    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(domain.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=domain, approx_grad=True)
            x_values.append(np.clip(result[0], *domain[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])

    def acquisition_function(self, x):
        """
        Compute the acquisition function.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f

        Returns
        ------
        af_value: float
            Value of the acquisition function at x
        """

        # TODO: enter your code here
        f_mean, f_std = self.f.predict(x.reshape(1, -1), return_std=True)  # gp sampling of f
        v_mean, v_std = self.v.predict(x.reshape(1, -1), return_std=True)  # gp sampling of v

        # means are a single value in a 2d array, therefore we extract them
        f_mean = f_mean[0]
        v_mean = v_mean[0]

        v_sat_prob = 1.0 - norm.cdf((SAFETY_THRESHOLD - v_mean) / v_std)  # probability of v(x) >= 1.2 being satisfied

        UCB = f_mean + UCB_BETA * f_std  # Upper Confidence Bound

        return UCB * v_sat_prob

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

        Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """

        # TODO: enter your code here
        x = x.squeeze(-1)
        self.prev_evals.append((x, f, v))  # add new evaluation

        xs = np.array([data[0] for data in self.prev_evals], dtype=float)
        fs = np.array([data[1] for data in self.prev_evals], dtype=float)
        vs = np.array([data[2] for data in self.prev_evals], dtype=float)

        self.f.fit(xs.reshape(-1, 1), fs)  # fit the GP of f with the new data
        self.v.fit(xs.reshape(-1, 1), vs)  # fit the GP of v with the new data

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        xs = np.array([data[0] for data in self.prev_evals], dtype=float)
        fs = np.array([data[1] for data in self.prev_evals], dtype=float).squeeze(-1)
        vs = np.array([data[2] for data in self.prev_evals], dtype=float).squeeze(-1)

        # return the x such that f(x) is the maximum possible and v(x) <= 1.2
        return xs[np.argmax(np.where(vs >= SAFETY_THRESHOLD, fs, -np.inf))]


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])


def f(x):
    """Dummy objective"""
    mid_point = domain[:, 0] + 0.5 * (domain[:, 1] - domain[:, 0])
    return - np.linalg.norm(x - mid_point, 2)  # -(x - 2.5)^2


def v(x):
    """Dummy speed"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*domain[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val > SAFETY_THRESHOLD]
    np.random.seed(SEED)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]
    return x_init



def main():
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)


    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x)
        cost_val = v(x)
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if v(solution) < 1.2:
        regret = 1
    else:
        regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()