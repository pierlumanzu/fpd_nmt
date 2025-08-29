import numpy as np
from itertools import chain, combinations
from pymoo.indicators.hv import HV
import time

from nsma.algorithms.gradient_based.gradient_based_algorithm import GradientBasedAlgorithm
from nsma.algorithms.genetic.genetic_utils.general_utils import calc_crowding_distance
from nsma.general_utils.pareto_utils import pareto_efficient

from direction_solvers.direction_solver_factory import DirectionSolverFactory
from line_searches.line_search_factory import LineSearchFactory
from problems.extended_problem import ExtendedProblem


class FPD(GradientBasedAlgorithm):

    def __init__(self,
                 max_iter: int, max_time: float, max_f_evals: int, max_g_evals: int,
                 verbose: bool, verbose_interspace: int,
                 plot_pareto_front: bool, plot_pareto_solutions: bool, plot_dpi: int,
                 theta_tol: float, qth_quantile: float,
                 gurobi_method: int, gurobi_verbose: bool,
                 ALS_alpha_0: float, ALS_delta: float, ALS_beta: float, ALS_min_alpha: float):

        GradientBasedAlgorithm.__init__(self,
                                        max_iter, max_time, max_f_evals,
                                        verbose, verbose_interspace,
                                        plot_pareto_front, plot_pareto_solutions, plot_dpi,
                                        theta_tol,
                                        True, gurobi_method, gurobi_verbose,
                                        ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha, name_ALS='BoundconstrainedFrontALS')

        self._direction_solver = DirectionSolverFactory.get_direction_calculator('P_DD', gurobi_method, gurobi_verbose)
        self._single_point_line_search = LineSearchFactory.get_line_search('MOALS', ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)

        self._qth_quantile = qth_quantile

        self.add_stopping_condition('max_g_evals', max_g_evals, 0, equal_required=True)
        
    def search(self, p_list: np.array, f_list: np.array, problem: ExtendedProblem):

        self.update_stopping_condition_current_value('max_time', time.time())

        efficient_points_idx = pareto_efficient(f_list)
        f_list = f_list[efficient_points_idx, :]
        p_list = p_list[efficient_points_idx, :]
        self.fpd_show_figure(p_list, f_list)

        for idx_p, p in enumerate(p_list):
            d_p, theta_p = self._direction_solver.compute_direction(problem, problem.evaluate_functions_jacobian(p), p)
            self.add_to_stopping_condition_current_value('max_g_evals', 1)

            d_list = d_p.reshape(1, -1) if idx_p == 0 else np.concatenate((d_list, d_p.reshape(1, -1)), axis=0)
            theta_list = np.array([theta_p]) if idx_p == 0 else np.concatenate((theta_list, np.array([theta_p])))

        crowding_quantile = np.inf 

        mem_dict = self.initialize_memory_information(p_list, f_list, d_list, theta_list)

        is_updated = True

        n_processed_points = 0
        sum_alpha = 0
        count_alpha = 0

        while not self.evaluate_stopping_conditions() and is_updated:

            is_updated = False

            self.output_data(f_list, 
                             n_g_evals= self.get_stopping_condition_current_value('max_g_evals'),
                             n_pp=n_processed_points,
                             ma=sum_alpha / count_alpha if count_alpha > 0 else np.inf)
            
            self.add_to_stopping_condition_current_value('max_iter', 1)

            self.fpd_show_figure(p_list, f_list, mem_dict=mem_dict)

            previous_p_list = np.copy(p_list)
            previous_f_list = np.copy(f_list)
            previous_d_list = np.copy(d_list)
            previous_theta_list = np.copy(theta_list)
                
            crowding_list = calc_crowding_distance(previous_f_list)
            is_finite_idx = np.isfinite(crowding_list)
            crowding_quantile = np.quantile(crowding_list[is_finite_idx], self._qth_quantile) if len(crowding_list[is_finite_idx]) > 0 else np.inf

            if len(p_list) == 1:
                sorted_idx = np.array([0])
            else:
                argmin_theta_list = np.argmin(theta_list)
                sorted_idx_crowding_list = np.flip(np.argsort(crowding_list))
                sorted_idx = np.concatenate((
                    np.array([argmin_theta_list]), 
                    np.delete(sorted_idx_crowding_list, np.where(sorted_idx_crowding_list == argmin_theta_list))
                ))

            previous_p_list = previous_p_list[sorted_idx, :]
            previous_f_list = previous_f_list[sorted_idx, :]
            previous_d_list = previous_d_list[sorted_idx, :]
            previous_theta_list = previous_theta_list[sorted_idx]
            crowding_list = crowding_list[sorted_idx]  

            p_list, f_list, d_list, theta_list = self.get_new_points_lists(p_list, f_list, d_list, theta_list, mem_dict)

            point_idx = 0

            while point_idx < len(previous_p_list):

                if self.evaluate_stopping_conditions():
                    break
                if self.exists_dominating_point(previous_f_list[point_idx, :], f_list):
                    point_idx += 1
                    continue

                x_p = previous_p_list[point_idx, :]
                f_p = previous_f_list[point_idx, :]
                
                d_sd, theta_sd = previous_d_list[point_idx, :], previous_theta_list[point_idx]

                is_refinement = False

                if not self.evaluate_stopping_conditions() and theta_sd < self._theta_tol:

                    n_processed_points += 1

                    if type(self) == FPD:
                        new_x_p, new_f_p, alpha, ls_f_evals = self._single_point_line_search.search(problem, x_p, f_p[np.newaxis], d_sd, theta_sd - 1/2 * np.linalg.norm(d_sd) ** 2)
                    else:
                        f_list_p = np.copy(f_list[~self.fast_non_dominated_filter(f_list, f_p[np.newaxis, :]), :])
                        new_x_p, new_f_p, alpha, ls_f_evals = self._single_point_line_search.search(problem, x_p, f_list_p if len(f_list_p) > 0 else f_p[np.newaxis], d_sd, theta_sd - 1/2 * np.linalg.norm(d_sd) ** 2)
                    
                    self.add_to_stopping_condition_current_value('max_f_evals', ls_f_evals)
                    sum_alpha += alpha
                    count_alpha += 1

                    if not self.evaluate_stopping_conditions() and new_x_p is not None:
                        
                        is_updated = True

                        is_refinement = True

                        J_p = problem.evaluate_functions_jacobian(new_x_p)
                        self.add_to_stopping_condition_current_value('max_g_evals', 1)

                        d_z, theta_z = self._direction_solver.compute_direction(problem, J_p, new_x_p)

                        efficient_points_idx = self.fast_non_dominated_filter(f_list, new_f_p[np.newaxis, :])

                        p_list = np.concatenate((p_list[efficient_points_idx, :],
                                                 new_x_p[np.newaxis, :]), axis=0)

                        f_list = np.concatenate((f_list[efficient_points_idx, :],
                                                 new_f_p[np.newaxis, :]), axis=0)

                        d_list = np.concatenate((d_list[efficient_points_idx, :],
                                                 d_z[np.newaxis, :]), axis=0)

                        theta_list = np.concatenate((theta_list[efficient_points_idx],
                                                     np.array([theta_z])))

                        x_p = np.copy(new_x_p)
                        f_p = np.copy(new_f_p)

                if not is_refinement:

                    J_p = problem.evaluate_functions_jacobian(x_p)
                    self.add_to_stopping_condition_current_value('max_g_evals', 1)

                    if type(self) != FPD:

                        efficient_points_idx = self.fast_non_dominated_filter(f_list, f_p[np.newaxis, :])

                        p_list = np.concatenate((p_list[efficient_points_idx, :],
                                                 x_p[np.newaxis, :]), axis=0)

                        f_list = np.concatenate((f_list[efficient_points_idx, :],
                                                 f_p[np.newaxis, :]), axis=0)

                        d_list = np.concatenate((d_list[efficient_points_idx, :],
                                                 d_sd[np.newaxis, :]), axis=0)

                        theta_list = np.concatenate((theta_list[efficient_points_idx],
                                                     np.array([theta_sd])))
               
                for I_k in self.objectives_powerset(problem.m):

                    if self.evaluate_stopping_conditions() or self.exists_dominating_point(f_p, f_list) or crowding_list[point_idx] < crowding_quantile:
                        break

                    partial_d_p, partial_theta_p = self._direction_solver.compute_direction(problem, J_p[I_k, ], x_p)

                    if not self.evaluate_stopping_conditions() and partial_theta_p < self._theta_tol:

                        n_processed_points += 1

                        new_x_p, new_f_p, alpha, ls_f_evals = self._line_search.search(problem, x_p, f_list, partial_d_p, 0., I=np.arange(problem.m))
                        self.add_to_stopping_condition_current_value('max_f_evals', ls_f_evals)
                        sum_alpha += alpha
                        count_alpha += 1

                        if not self.evaluate_stopping_conditions() and new_x_p is not None:

                            is_updated = True

                            efficient_points_idx = self.fast_non_dominated_filter(f_list, new_f_p.reshape(1, -1))

                            p_list = np.concatenate((p_list[efficient_points_idx, :],
                                                     new_x_p.reshape(1, -1)), axis=0)

                            f_list = np.concatenate((f_list[efficient_points_idx, :],
                                                     new_f_p.reshape(1, -1)), axis=0)

                            d_p, theta_p = self._direction_solver.compute_direction(problem, problem.evaluate_functions_jacobian(new_x_p), new_x_p)
                            self.add_to_stopping_condition_current_value('max_g_evals', 1)
                            
                            d_list = np.concatenate((d_list[efficient_points_idx, :], d_p.reshape(1, -1)), axis=0)
                            theta_list = np.concatenate((theta_list[efficient_points_idx], np.array([theta_p])))

                point_idx += 1

            if type(self) != FPD and point_idx < len(previous_p_list):
                
                for pidx in range(point_idx, len(previous_p_list)):

                    if not self.exists_dominating_point(previous_f_list[pidx, :], f_list):

                        efficient_points_idx = self.fast_non_dominated_filter(f_list, previous_f_list[pidx, :].reshape(1, -1))

                        p_list = np.concatenate((p_list[efficient_points_idx, :],
                                                 previous_p_list[pidx, :].reshape((1, problem.n))), axis=0)

                        f_list = np.concatenate((f_list[efficient_points_idx, :],
                                                 previous_f_list[pidx, :].reshape((1, problem.m))), axis=0)

                        d_list = np.concatenate((d_list[efficient_points_idx, :],
                                                 previous_d_list[pidx, :].reshape((1, problem.n))), axis=0)

                        theta_list = np.concatenate((theta_list[efficient_points_idx],
                                                     np.array([previous_theta_list[pidx]])))

            p_list, f_list, d_list, theta_list, mem_dict = self.update_memory_information(p_list, f_list, d_list, theta_list, mem_dict)

            self.fpd_show_figure(p_list, f_list, mem_dict=mem_dict)

        self.output_data(f_list,
                         n_g_evals= self.get_stopping_condition_current_value('max_g_evals'),
                         n_pp=n_processed_points,
                         ma=sum_alpha / count_alpha if count_alpha > 0 else np.inf)
        self.close_figure()

        return (p_list,
                f_list,
                {   
                    "Ni": self.get_stopping_condition_current_value('max_iter'),
                    "T": time.time() - self.get_stopping_condition_current_value('max_time'),
                    "Nf": self.get_stopping_condition_current_value('max_f_evals'),
                    "Ng": self.get_stopping_condition_current_value('max_g_evals'),
                    "n_pp": n_processed_points,
                    "ma": sum_alpha / count_alpha if count_alpha > 0 else np.inf
                })

    @staticmethod
    def objectives_powerset(m: int):
        s = list(range(m))        
        return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s))))

    @staticmethod
    def exists_dominating_point(f: np.array, f_list: np.array):

        if np.isnan(f).any():
            return True

        n_obj = len(f)

        f = np.reshape(f, (1, n_obj))
        dominance_matrix = f_list - f

        return (np.logical_and(np.sum(dominance_matrix <= 0, axis=1) == n_obj, np.sum(dominance_matrix < 0, axis=1) > 0)).any()

    @staticmethod
    def fast_non_dominated_filter(curr_f_list: np.array, new_f_list: np.array):

        n_new_points, m = new_f_list.shape
        efficient = np.array([True] * curr_f_list.shape[0])

        for i in range(n_new_points):
            dominance_matrix = curr_f_list - np.reshape(new_f_list[i, :], newshape=(1, m))
            dominated_idx = np.sum(dominance_matrix >= 0, axis=1) == m

            assert len(dominated_idx.shape) == 1
            dom_indices = np.where(dominated_idx)[0]

            if len(dom_indices) > 0:
                efficient[dom_indices] = False

        return efficient

    def fpd_show_figure(self, p_list: np.array, f_list: np.array, mem_dict: dict = None):
        return super().show_figure(p_list, f_list)

    def initialize_memory_information(self, p_list: np.array, f_list: np.array, d_list: np.array, theta_list: np.array):
        return {"previous_f_list": np.copy(f_list)}

    def get_new_points_lists(self, p_list: np.array, f_list: np.array, d_list: np.array, theta_list: np.array, mem_dict):
        return p_list, f_list, d_list, theta_list

    def update_memory_information(self, p_list: np.array, f_list: np.array, d_list: np.array, theta_list: np.array, mem_dict):
        mem_dict["previous_f_list"] = np.copy(f_list)
        return p_list, f_list, d_list, theta_list, mem_dict