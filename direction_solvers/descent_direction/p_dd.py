import numpy as np
from gurobipy import Model, GRB, GurobiError

from nsma.direction_solvers.descent_direction.dds import DDS
from nsma.direction_solvers.gurobi_settings import GurobiSettings

from problems.extended_problem import ExtendedProblem


class P_DD(DDS, GurobiSettings):

    def __init__(self, gurobi_method: int, gurobi_verbose: bool):
        DDS.__init__(self)
        GurobiSettings.__init__(self, gurobi_method, gurobi_verbose)

    def compute_direction(self, problem: ExtendedProblem, Jac: np.array, x_p: np.array = None):
        assert x_p is not None

        m, n = Jac.shape

        if np.isinf(Jac).any() or np.isnan(Jac).any():
            return np.zeros(n), 0

        try:
            model = Model("Projected Descent Direction")

            model.setParam("OutputFlag", self._gurobi_verbose)
            model.setParam("Method", self._gurobi_method)

            z = model.addMVar(n, lb=problem.lb, ub=problem.ub, name="z")
            beta = model.addMVar(1, lb=-np.inf, ub=0., name="beta")

            obj = beta + 0.5 * (z - x_p) @ (z - x_p)
            model.setObjective(obj)

            for j in range(m):
                model.addConstr(Jac[j, :] @ (z - x_p) <= beta, name='Jacobian constraint n°{}'.format(j))

            model.update()

            for i in range(n):
                z[i].start = float(x_p[i])
            beta.start = 0.

            model.optimize()

            if model.Status == GRB.OPTIMAL:
                return z.x - x_p, model.getObjective().getValue()
            else:
                return np.zeros(n), 0

        except GurobiError:
            if self._gurobi_verbose:
                print('Gurobi Error')
            return np.zeros(n), 0

        except AttributeError:
            if self._gurobi_verbose:
                print('Attribute Error')
            return np.zeros(n), 0
