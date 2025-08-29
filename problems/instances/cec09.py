import numpy as np
import tensorflow as tf

from problems.extended_problem import ExtendedProblem


class CEC(ExtendedProblem):

    def __init__(self, n):
        ExtendedProblem.__init__(self, n)

    @staticmethod
    def family_name():
        return 'CEC'


class CEC09_2(CEC):

    def __init__(self, n):
        assert n >= 3
        CEC.__init__(self, n)

        J1 = np.arange(2, self.n, 2)
        J2 = np.arange(1, self.n, 2)

        y_odd = 2 * tf.reduce_sum([(self._z[j] - (0.3 * self._z[0] ** 2 * tf.cos(24 * np.pi * self._z[0] + 4 * (j + 1) * np.pi / n) + 0.6 * self._z[0]) * tf.cos(6 * np.pi * self._z[0] + (j + 1) * np.pi / n)) ** 2 for j in J1]) / len(J1)
        y_even = 2 * tf.reduce_sum([(self._z[j] - (0.3 * self._z[0] ** 2 * tf.cos(24 * np.pi * self._z[0] + 4 * (j + 1) * np.pi / n) + 0.6 * self._z[0]) * tf.sin(6 * np.pi * self._z[0] + (j + 1) * np.pi / n)) ** 2 for j in J2]) / len(J2)

        self.set_objectives([
            self._z[0] + y_odd,
            1 - tf.sqrt(self._z[0]) + y_even
        ])

        lb = -1 * np.ones(n)
        lb[0] = 0.0
        self.filtered_lb_for_ini = np.copy(lb)
        self.set_lb(np.copy(lb))

        self.filtered_ub_for_ini = np.ones(n)
        self.set_ub(np.ones(n))

    @staticmethod
    def name():
        return 'CEC09_2'
