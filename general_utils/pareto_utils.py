import numpy as np
import scipy

from nsma.general_utils.pareto_utils import pareto_efficient

from problems.extended_problem import ExtendedProblem


def points_postprocessing(p_list: np.array, f_list: np.array, problem: ExtendedProblem):
    assert len(p_list) == len(f_list)
    old_n_points, _ = p_list.shape

    print('Found {} points'.format(old_n_points))

    for p in range(old_n_points):
        f_list[p, :] = problem.evaluate_functions(p_list[p, :])

    p_list, f_list = remove_duplicates_point(p_list, f_list)

    n_points, _ = p_list.shape

    if old_n_points - n_points > 0:
        print('Warning: found {} duplicate points'.format(old_n_points - n_points))

    feasible = [True] * n_points
    infeasible_points = 0
    for p in range(n_points):
        if not problem.check_point_feasibility(p_list[p, :]):
            feasible[p] = False
            infeasible_points += 1
    if infeasible_points > 0:
        print('Warning: found {} infeasible points'.format(infeasible_points))

    p_list = p_list[feasible, :]
    f_list = f_list[feasible, :]

    old_n_points, _ = p_list.shape

    efficient_points_idx = pareto_efficient(f_list)
    p_list = p_list[efficient_points_idx, :]
    f_list = f_list[efficient_points_idx, :]

    if old_n_points - len(p_list) > 0:
        print('Warning: found {} dominated points'.format(old_n_points - len(p_list)))
    print('Result: found {} non-dominated points'.format(len(p_list)))
    print()

    return p_list, f_list


def remove_duplicates_point(p_list: np.array, f_list: np.array):

    is_duplicate = np.array([False] * p_list.shape[0])

    D = scipy.spatial.distance.cdist(p_list, p_list)
    D[np.triu_indices(len(p_list))] = np.inf

    D[np.isnan(D)] = np.inf

    is_duplicate[np.any(D < 1e-16, axis=1)] = True

    p_list = p_list[~is_duplicate]
    f_list = f_list[~is_duplicate]

    return p_list, f_list
