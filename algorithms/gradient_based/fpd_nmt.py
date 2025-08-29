import numpy as np
from pymoo.indicators.hv import HV

from nsma.general_utils.pareto_utils import pareto_efficient

from algorithms.gradient_based.fpd import FPD


class FPD_NMT(FPD):

    def __init__(self,
                 max_iter: int, max_time: float, max_f_evals: int, max_g_evals: int,
                 verbose: bool, verbose_interspace: int,
                 plot_pareto_front: bool, plot_pareto_solutions: bool, plot_dpi: int,
                 theta_tol: float, qth_quantile: float, M: int,
                 gurobi_method: int, gurobi_verbose: bool,
                 ALS_alpha_0: float, ALS_delta: float, ALS_beta: float, ALS_min_alpha: float):

        FPD.__init__(self,
                     max_iter, max_time, max_f_evals, max_g_evals,
                     verbose, verbose_interspace,
                     plot_pareto_front, plot_pareto_solutions, plot_dpi,
                     theta_tol, qth_quantile,
                     gurobi_method, gurobi_verbose,
                     ALS_alpha_0, ALS_delta, ALS_beta, ALS_min_alpha)

        self._M = M

    def fpd_show_figure(self, p_list: np.array, f_list: np.array, mem_dict: dict = None):
        if mem_dict is not None:
            return super().show_figure(p_list, f_list, fh_list=mem_dict["C_f"])
        else:
            return super().show_figure(p_list, f_list)

    def initialize_memory_information(self, p_list: np.array, f_list: np.array, d_list: np.array, theta_list: np.array):
        
        mem_dict = {
            "p_memory": [np.copy(p_list)],
            "f_memory": [np.copy(f_list)],
            "d_memory": [np.copy(d_list)],
            "theta_memory": [np.copy(theta_list)],
            "nadir_point": np.amax(f_list, axis=0)
        }
        
        mem_dict["HV_memory"] = np.array([HV(ref_point=mem_dict["nadir_point"])(f_list)])
        
        mem_dict["C_p"] = np.copy(mem_dict["p_memory"][0])
        mem_dict["C_f"] = np.copy(mem_dict["f_memory"][0])
        mem_dict["C_d"] = np.copy(mem_dict["d_memory"][0])
        mem_dict["C_theta"] = np.copy(mem_dict["theta_memory"][0])
        
        mem_dict["C_memory"] = [np.copy(mem_dict["C_f"])]
            
        return mem_dict

    def get_new_points_lists(self, p_list: np.array, f_list: np.array, d_list: np.array, theta_list: np.array, mem_dict):
        return super().get_new_points_lists(
            np.copy(mem_dict["C_p"]),
            np.copy(mem_dict["C_f"]),
            np.copy(mem_dict["C_d"]),
            np.copy(mem_dict["C_theta"]),
            mem_dict
        )

    def update_memory_information(self, p_list: np.array, f_list: np.array, d_list: np.array, theta_list: np.array, mem_dict):
        
        if len(mem_dict["p_memory"]) == self._M + 1:
            mem_dict["p_memory"] = mem_dict["p_memory"][1:]
            mem_dict["f_memory"] = mem_dict["f_memory"][1:]
            mem_dict["d_memory"] = mem_dict["d_memory"][1:]
            mem_dict["theta_memory"] = mem_dict["theta_memory"][1:]
            mem_dict["HV_memory"] = mem_dict["HV_memory"][1:]
        
        new_nadir_point = np.amax(np.concatenate((mem_dict["nadir_point"][np.newaxis], f_list), axis=0), axis=0)
        if not np.allclose(mem_dict["nadir_point"], new_nadir_point):
            mem_dict["nadir_point"] = new_nadir_point
            mem_dict["HV_memory"] = np.array([HV(ref_point=mem_dict["nadir_point"])(mem_dict["f_memory"][i]) for i in range(len(mem_dict["f_memory"]))])
        mem_dict["HV_memory"] = np.concatenate((mem_dict["HV_memory"], np.array([HV(ref_point=mem_dict["nadir_point"])(f_list)])))

        argmin_hv = np.argmin(mem_dict["HV_memory"])

        if argmin_hv < len(mem_dict["p_memory"]):
            p_list = np.concatenate((mem_dict["p_memory"][argmin_hv], p_list), axis=0)
            f_list = np.concatenate((mem_dict["f_memory"][argmin_hv], f_list), axis=0)
            d_list = np.concatenate((mem_dict["d_memory"][argmin_hv], d_list), axis=0)
            theta_list = np.concatenate((mem_dict["theta_memory"][argmin_hv], theta_list), axis=0)

            efficient_points_idx = pareto_efficient(f_list)
            p_list = p_list[efficient_points_idx]
            f_list = f_list[efficient_points_idx]
            d_list = d_list[efficient_points_idx]   
            theta_list = theta_list[efficient_points_idx]

            mem_dict["HV_memory"][-1] = HV(ref_point=mem_dict["nadir_point"])(f_list)

        mem_dict["p_memory"].append(np.copy(p_list))
        mem_dict["f_memory"].append(np.copy(f_list))
        mem_dict["d_memory"].append(np.copy(d_list))
        mem_dict["theta_memory"].append(np.copy(theta_list))

        mem_dict["C_p"] = np.concatenate((mem_dict["C_p"], np.copy(mem_dict["p_memory"][argmin_hv])), axis=0)
        mem_dict["C_f"] = np.concatenate((mem_dict["C_f"], np.copy(mem_dict["f_memory"][argmin_hv])), axis=0)
        mem_dict["C_d"] = np.concatenate((mem_dict["C_d"], np.copy(mem_dict["d_memory"][argmin_hv])), axis=0)
        mem_dict["C_theta"] = np.concatenate((mem_dict["C_theta"], np.copy(mem_dict["theta_memory"][argmin_hv])))
        
        efficient_points_idx = pareto_efficient(mem_dict["C_f"])
        mem_dict["C_p"] = mem_dict["C_p"][efficient_points_idx]
        mem_dict["C_f"] = mem_dict["C_f"][efficient_points_idx]
        mem_dict["C_d"] = mem_dict["C_d"][efficient_points_idx]
        mem_dict["C_theta"] = mem_dict["C_theta"][efficient_points_idx]

        efficient_points_idx = self.fast_non_dominated_filter(-f_list, -mem_dict["C_f"])
        mem_dict["C_p"] = np.concatenate((mem_dict["C_p"], p_list[efficient_points_idx]), axis=0)
        mem_dict["C_f"] = np.concatenate((mem_dict["C_f"], f_list[efficient_points_idx]), axis=0)
        mem_dict["C_d"] = np.concatenate((mem_dict["C_d"], d_list[efficient_points_idx]), axis=0)
        mem_dict["C_theta"] = np.concatenate((mem_dict["C_theta"], theta_list[efficient_points_idx]))
        
        mem_dict["C_memory"].append(np.copy(mem_dict["C_f"]))

        if len(mem_dict["C_memory"]) > self._M + 1:
            mem_dict["C_memory"] = mem_dict["C_memory"][1:]

        return p_list, f_list, d_list, theta_list, mem_dict
