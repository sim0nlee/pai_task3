import numpy as np
import os
import random
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern

domain = np.array([[0, 5]])
n_dim = 1

""" Solution """


class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration. """
        # TODO: enter your code here
        self.prev_evals = []
        self.f = GaussianProcessRegressor(
            kernel=0.5 * Matern(length_scale=0.5, nu=2.5) + WhiteKernel(0.15 ** 2),
            optimizer=None,
            normalize_y=True)
        self.v = GaussianProcessRegressor(
            kernel=1.5 + np.sqrt(2) * Matern(length_scale=0.5, nu=2.5) + WhiteKernel(0.0001 ** 2),
            optimizer=None,
            normalize_y=True
        )

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """

        # TODO: enter your code here
        # In implementing this function, you may use optimize_acquisition_function() defined below.
        if not self.prev_evals:
            nxt = np.array([[np.random.uniform(0, 5)]])
        else:
            nxt = self.optimize_acquisition_function()

        return nxt


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
            f_values.append(result[1])

        ind = np.argmin(f_values)
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
        # phi = norm.cdf
        # N = norm.pdf
        #
        # speed_mean, speed_std = self.speed.predict(x.reshape(-1, 1), return_std=True)
        # speed_sat_prob = 1 - phi((1.2 - speed_mean) / speed_std)
        #
        # f_min = min([data[1] for data in self.points])
        # obj_mean, obj_std = self.objective.predict(x.reshape(-1, 1), return_std=True)
        #
        # return ((obj_mean - f_min) * (phi((f_min - obj_mean) / obj_std)) + N(
        #     (f_min - obj_mean) / obj_std)) * speed_sat_prob
        con_mean, con_std = self.v.predict(x.reshape(1, -1), return_std=True)
        con_prob_x = 1.0 - norm.cdf(1.2, con_mean, con_std)  # Proba of x satisfying constraint v^(x) >= 1.2

        f_min = np.min([triplet[1] for triplet in self.prev_evals])  # Get current min fct value

        obj_mean, obj_std = self.f.predict(x.reshape(1, -1), return_std=True)
        z = (f_min - obj_mean) / obj_std  # Standardized
        ei_x = obj_std * (z * norm.cdf(z) + norm.pdf(z))  # Expected improvement for obs x
        af_x = ei_x * con_prob_x  # Acquisition function
        return af_x


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
        self.prev_evals.append([float(x), float(f), float(v)])

        x_vals = np.array([data[0] for data in self.prev_evals])
        f_vals = np.array([data[1] for data in self.prev_evals])
        v_vals = np.array([data[2] for data in self.prev_evals])

        self.f.fit(x_vals.reshape(-1, 1), f_vals)
        self.v.fit(x_vals.reshape(-1, 1), v_vals)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """

        # TODO: enter your code here
        # x_vals = np.array([data[0] for data in self.prev_evals])
        # f_vals = np.array([data[1] for data in self.prev_evals])
        # v_vals = np.array([data[2] for data in self.prev_evals])
        #
        # satisfied = np.argwhere(v_vals >= 1.2)
        #
        # if satisfied.size == 0:
        #     print('No solution satisfying the speed constraint')
        #     print()
        #
        # return x_vals[0] if satisfied.size == 0 else x_vals[np.argmin(f_vals[satisfied])]
        x_vals = np.array([triplet[0] for triplet in self.prev_evals])
        obj_vals = np.array([triplet[1] for triplet in self.prev_evals])
        con_vals = np.array([triplet[2] for triplet in self.prev_evals])

        sorted_inds = np.argsort(obj_vals)
        feasible = False
        argmin_ind = 0
        for ind in sorted_inds:
            if con_vals[ind] >= 1.2:
                argmin_ind = ind
                feasible = True
                break

        # if not feasible:
        #     print('Feasibility notifier: No feasible solution found!')
        # print('Minimiser triplet: ', (x_vals[argmin_ind], obj_vals[argmin_ind], con_vals[argmin_ind]))
        # print('Number of iterations: ', len(self.prev_evals))

        return x_vals[argmin_ind]


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


def main():
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = domain[:, 0] + (domain[:, 1] - domain[:, 0]) * np.random.rand(
        1, n_dim)
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
