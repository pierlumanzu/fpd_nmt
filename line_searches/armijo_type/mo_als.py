import numpy as np

from nsma.line_searches.armijo_type.als import ALS

from problems.extended_problem import ExtendedProblem


class MOALS(ALS):

    def __init__(self, alpha_0: float, delta: float, beta: float, min_alpha: float):
        ALS.__init__(self, alpha_0, delta, beta, min_alpha)

    def search(self, problem: ExtendedProblem, x: np.array, f_list: np.array, d: np.array, theta: float, I: np.array = None):
        assert I is None

        alpha = self._alpha_0
        new_x = x + alpha * d
        new_f = problem.evaluate_functions(new_x)
        f_eval = 1

        while (not problem.check_point_feasibility(new_x) or
               np.isnan(new_f).any() or
               np.isinf(new_f).any() or
               len(np.where(np.sum(new_f >= f_list + self._beta * alpha * theta, axis=1) > 0)[0]) == len(f_list)) and alpha > self._min_alpha:
            alpha *= self._delta
            new_x = x + alpha * d
            new_f = problem.evaluate_functions(new_x)
            f_eval += 1

        if alpha <= self._min_alpha:
            alpha = 0
            return None, None, alpha, f_eval

        return new_x, new_f, alpha, f_eval