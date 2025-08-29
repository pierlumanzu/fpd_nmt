from direction_solvers.descent_direction.p_dd import P_DD


class DirectionSolverFactory:

    @staticmethod
    def get_direction_calculator(direction_type: str, gurobi_method: int, gurobi_verbose: bool):

        if direction_type == 'P_DD':
            return P_DD(gurobi_method, gurobi_verbose)

        raise NotImplementedError
