import argparse
import sys


def get_args():

    parser = argparse.ArgumentParser(description='FD Framework')

    parser.add_argument('--algs', type=str, help='Algorithms', nargs='+', default=['FPD', 'FPD_NMT'])

    parser.add_argument('--probs', help='Problems to evaluate', nargs='+', default=['CEC'])

    parser.add_argument('--max_iter', help='Maximum number of iterations', default=None, type=int)

    parser.add_argument('--max_time', help='Maximum number of elapsed minutes per problem', default=None, type=float)

    parser.add_argument('--max_f_evals', help='Maximum number of function evaluations', default=None, type=int)

    parser.add_argument('--max_g_evals', help='Maximum number of gradient evaluations', default=None, type=int)

    parser.add_argument('--verbose', help='Verbose during the iterations', action='store_true', default=False)

    parser.add_argument('--verbose_interspace', help='Used interspace in the verbose (Requirements: verbose activated)', default=20, type=int)

    parser.add_argument('--plot_pareto_front', help='Plot Pareto front', action='store_true', default=False)

    parser.add_argument('--plot_pareto_solutions', help='Plot Pareto solutions (Requirements: plot_pareto_front activated; n in [2, 3])', action='store_true', default=False)

    parser.add_argument('--general_export', help='Export fronts (including plots), execution times and arguments files', action='store_true', default=False)

    parser.add_argument('--export_pareto_solutions', help='Export pareto solutions, including the plots if n in [2, 3] (Requirements: general_export activated)', action='store_true', default=False)

    parser.add_argument('--plot_dpi', help='DPI of the saved plots (Requirements: general_export activated)', default=100, type=int)

    ####################################################
    ### FD ###
    ####################################################

    parser.add_argument('--FD_theta_tol', help='FD parameter -- theta tolerance', default=-1.0e-7, type=float)

    parser.add_argument('--FD_qth_quantile', help='FD parameter -- q-th quantile', default=0.95, type=float)

    parser.add_argument('--FD_M', help='FD parameter -- Memory for non-monotone', default=[2, 4, 20], type=int, nargs='+')
 
    ####################################################
    ### Gurobi ###
    ####################################################

    parser.add_argument('--gurobi_method', help='Gurobi parameter -- used method', default=1, type=int)

    parser.add_argument('--gurobi_verbose', help='Gurobi parameter -- verbose during the Gurobi iterations', action='store_true', default=False)

    ####################################################
    ### Armijo-type Line Search ###
    ####################################################

    parser.add_argument('--ALS_alpha_0', help='ALS parameter -- initial step size', default=1, type=float)

    parser.add_argument('--ALS_delta', help='ALS parameter -- coefficient for step size contraction', default=0.5, type=float)

    parser.add_argument('--ALS_beta', help='ALS parameter -- beta', default=1.0e-4, type=float)

    parser.add_argument('--ALS_min_alpha', help='ALS parameter -- min alpha', default=1.0e-7, type=float)

    return parser.parse_args(sys.argv[1:])

