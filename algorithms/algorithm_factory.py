from algorithms.gradient_based.fpd import FPD
from algorithms.gradient_based.fpd_nmt import FPD_NMT


class AlgorithmFactory:

    @staticmethod
    def get_algorithm(algorithm_name: str, **kwargs):

        general_settings = kwargs['general_settings']
        algorithms_settings = kwargs['algorithms_settings']

        FD_settings = algorithms_settings['FD']
        DDS_settings = kwargs['DDS_settings']
        ALS_settings = kwargs['ALS_settings']

        if algorithm_name == 'FPD':
            return FPD(general_settings['max_iter'],
                       general_settings['max_time'],
                       general_settings['max_f_evals'],
                       general_settings['max_g_evals'],
                       general_settings['verbose'],
                       general_settings['verbose_interspace'],
                       general_settings['plot_pareto_front'],
                       general_settings['plot_pareto_solutions'],
                       general_settings['plot_dpi'],
                       FD_settings['theta_tol'],
                       FD_settings['qth_quantile'],
                       DDS_settings['gurobi_method'],
                       DDS_settings['gurobi_verbose'],
                       ALS_settings['alpha_0'],
                       ALS_settings['delta'],
                       ALS_settings['beta'],
                       ALS_settings['min_alpha'])

        elif algorithm_name == 'FPD_NMT':
            return FPD_NMT(general_settings['max_iter'],
                           general_settings['max_time'],
                           general_settings['max_f_evals'],
                           general_settings['max_g_evals'],
                           general_settings['verbose'],
                           general_settings['verbose_interspace'],
                           general_settings['plot_pareto_front'],
                           general_settings['plot_pareto_solutions'],
                           general_settings['plot_dpi'],
                           FD_settings['theta_tol'],
                           FD_settings['qth_quantile'],
                           kwargs['M'],
                           DDS_settings['gurobi_method'],
                           DDS_settings['gurobi_verbose'],
                           ALS_settings['alpha_0'],
                           ALS_settings['delta'],
                           ALS_settings['beta'],
                           ALS_settings['min_alpha'])

        raise NotImplementedError
